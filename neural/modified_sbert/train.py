import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, datasets, evaluation, LoggingHandler, SentenceTransformer, util
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

import logging
from transformers import logging as lg

from modified_sbert import data_loaders, clu_evaluators


lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """
    Source: https://github.com/UKPLab/sentence-transformers/issues/1604
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """

    def __init__(self, model, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sentence_features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = self.model(sentence_features[0])['sentence_embedding']

        #Nils: Normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        ## Nils: Add n_views dimension
        features = torch.unsqueeze(features, 1)

        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def train_biencoder(
        train_data: dict = None,
        dev_data: dict = None,
        base_model='sentence-transformers/all-MiniLM-L12-v2',
        add_pooling_layer=False,
        train_batch_size=64,
        num_epochs=10,
        warmup_epochs=1,
        loss_fn='contrastive',
        loss_params=None,
        model_save_path="output",
):

    os.makedirs(model_save_path, exist_ok=True)

    # Base language model
    if add_pooling_layer:
        word_embedding_model = models.Transformer(base_model, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(base_model)

    # Loss functions
    if loss_fn == "contrastive":
        train_loss = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=loss_params['distance_metric'],
            margin=loss_params['margin']
        )

        train_samples = data_loaders.load_data_as_pairs(train_data, type="neural")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "cosine":
        train_loss = losses.CosineSimilarityLoss(model=model)

        train_samples = data_loaders.load_data_as_pairs(train_data, type="neural")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "triplet":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=loss_params['distance_metric'],
            triplet_margin=loss_params['margin']
        )

        train_samples = data_loaders.load_data_as_triplets(train_data, type="neural")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        train_samples = data_loaders.load_data_as_triplets(train_data, type="neural")

        # Special dataloader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    elif loss_fn == "supcon":
        train_loss = losses.SupConLoss(model=model)

        train_samples = data_loaders.load_data_as_individuals(train_data, type="neural")

        # Special dataset "SentenceLabelDataset" to wrap out train_set
        # It yields batches that contain at least two samples with the same label
        train_data_sampler = SentenceLabelDataset(train_samples)
        train_dataloader = DataLoader(train_data_sampler, batch_size=train_batch_size)

    # Evaluate with multiple evaluators
    dev_pairs = data_loaders.load_data_as_pairs(dev_data, type="dev")
    # dev_triplets = data_loaders.load_data_as_triplets(dev_data, type="dev")

    evaluators = [
        evaluation.BinaryClassificationEvaluator.from_input_examples(dev_pairs),
        # evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_pairs),
        # evaluation.TripletEvaluator.from_input_examples(dev_triplets),
        clu_evaluators.ClusterEvaluator.from_input_examples(dev_pairs, cluster_type="agglomerative")
    ]

    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    logger.info("Evaluate model without neural")
    seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=seq_evaluator,
        epochs=num_epochs,
        warmup_steps=math.ceil(len(train_dataloader) * warmup_epochs),
        output_path=model_save_path,
        evaluation_steps=112,
        checkpoint_save_steps=112,
        checkpoint_path=model_save_path,
        save_best_model=True,
        checkpoint_save_total_limit=10
    )


def train_crossencoder(
        train_data,
        dev_data,
        model_name,
        lr,
        train_batch_size,
        num_epochs,
        warm_up_perc,
        eval_per_epoch,
        model_save_path,
):

    model = CrossEncoder(model_name, num_labels=1)

    train = data_loaders.load_data_as_pairs(train_data, type="neural")
    dev = data_loaders.load_data_as_pairs(dev_data, type="dev")

    # Wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

    # Evaluate with multiple evaluators
    evaluators = [
        CEBinaryClassificationEvaluator.from_input_examples(dev, name='dev'),
        clu_evaluators.CEClusterEvaluator.from_input_examples(dev, name='dev'),
    ]

    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warm_up_perc)
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=seq_evaluator,
              epochs=num_epochs,
              evaluation_steps=int(len(train_dataloader)*(1/eval_per_epoch)),
              loss_fct=torch.nn.BCEWithLogitsLoss(),
              optimizer_params={"lr": lr},
              warmup_steps=warmup_steps,
              output_path=model_save_path)
