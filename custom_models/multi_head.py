import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from transformers import RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import (TokenClassifierOutput,SequenceClassifierOutput)
from torch.utils.data import DataLoader, Dataset
import numpy
import os
import pandas
import sys
import tempfile
import transformers
import argparse
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead, RobertaLayer, RobertaPooler

##################################################################################################################################################################################
##################################################################################################################################################################################

class RobertaClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()

        self.extra_transformer_layer_1 = RobertaLayer(config)
        self.extra_transformer_layer_2 = RobertaLayer(config)
        self.extra_transformer_layer_3 = RobertaLayer(config)
        # self.extra_transformer_layer_4 = RobertaLayer(config)
        # self.extra_transformer_layer_5 = RobertaLayer(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features    #[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.extra_transformer_layer_1(x)
        x = self.extra_transformer_layer_2(x[0])
        x = self.extra_transformer_layer_3(x[0])
        # x = self.extra_transformer_layer_4(x[0])
        # x = self.extra_transformer_layer_5(x[0])
        x = x[0][:,0,:]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
##################################################################################################################################################################################
##################################################################################################################################################################################

class MultiHead(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.language_heads = nn.ModuleDict({
            'EN': RobertaClassificationHead(config),
            'EL': RobertaClassificationHead(config),
            'DE': RobertaClassificationHead(config),
            'TR': RobertaClassificationHead(config),
            'FR': RobertaClassificationHead(config),
            'BG': RobertaClassificationHead(config),
            'HE': RobertaClassificationHead(config),
            'IT': RobertaClassificationHead(config),
            'NL': RobertaClassificationHead(config),
        })

        # self.classifier = RobertaClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language=None) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output = self.bilstm(outputs[0])[0]

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        id2lang_dict={0: 'EN', 1: 'EL', 2: 'DE', 3: 'TR', 4: 'FR', 5: 'BG', 6: 'HE', 7: 'IT', 8: 'NL'}
        
        head_1=self.language_heads[id2lang_dict[int(language[0])]]
        head_2=self.language_heads[id2lang_dict[int(language[1])]]
        head_3=self.language_heads[id2lang_dict[int(language[2])]]
        head_4=self.language_heads[id2lang_dict[int(language[3])]]
        head_5=self.language_heads[id2lang_dict[int(language[4])]]
        head_6=self.language_heads[id2lang_dict[int(language[5])]]
        head_7=self.language_heads[id2lang_dict[int(language[6])]]
        head_8=self.language_heads[id2lang_dict[int(language[7])]]
        
        # language_head = self.language_heads[id2lang_dict[int(language)]]
        # id2lang_dict={0: 'EN', 1: 'EL', 2: 'DE', 3: 'TR', 4: 'FR', 5: 'BG', 6: 'HE', 7: 'IT', 8: 'NL'}
        # lang_index = language.item()
        # logits = language_head(outputs.last_hidden_state)

        split_tensors = torch.split(outputs.last_hidden_state, split_size_or_sections=1, dim=0)

        logits_1 = head_1(split_tensors[0])
        logits_2 = head_2(split_tensors[1])
        logits_3 = head_3(split_tensors[2])
        logits_4 = head_4(split_tensors[3])
        logits_5 = head_5(split_tensors[4])
        logits_6 = head_6(split_tensors[5])
        logits_7 = head_7(split_tensors[6])
        logits_8 = head_8(split_tensors[7])

        logits=torch.cat((logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8), dim=0)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

##################################################################################################################################################################################
##################################################################################################################################################################################



##################################################################################################################################################################################
##################################################################################################################################################################################

class MultiHead_MultiLabel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = class_weights
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.language_heads = nn.ModuleDict({
            'EN': RobertaClassificationHeadCustom(config),
            'EL': RobertaClassificationHeadCustom(config),
            'DE': RobertaClassificationHeadCustom(config),
            'TR': RobertaClassificationHeadCustom(config),
            'FR': RobertaClassificationHeadCustom(config),
            'BG': RobertaClassificationHeadCustom(config),
            'HE': RobertaClassificationHeadCustom(config),
            'IT': RobertaClassificationHeadCustom(config),
            'NL': RobertaClassificationHeadCustom(config),
        })

        # self.classifier = RobertaClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language=None) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output = self.bilstm(outputs[0])[0]

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        id2lang_dict={0: 'EN', 1: 'EL', 2: 'DE', 3: 'TR', 4: 'FR', 5: 'BG', 6: 'HE', 7: 'IT', 8: 'NL'}
        batch_sz=len(language)

        if batch_sz==8:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
            head_4=self.language_heads[id2lang_dict[int(language[3])]]
            head_5=self.language_heads[id2lang_dict[int(language[4])]]
            head_6=self.language_heads[id2lang_dict[int(language[5])]]
            head_7=self.language_heads[id2lang_dict[int(language[6])]]
            head_8=self.language_heads[id2lang_dict[int(language[7])]]
        elif batch_sz==7:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
            head_4=self.language_heads[id2lang_dict[int(language[3])]]
            head_5=self.language_heads[id2lang_dict[int(language[4])]]
            head_6=self.language_heads[id2lang_dict[int(language[5])]]
            head_7=self.language_heads[id2lang_dict[int(language[6])]]
        elif batch_sz==6:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
            head_4=self.language_heads[id2lang_dict[int(language[3])]]
            head_5=self.language_heads[id2lang_dict[int(language[4])]]
            head_6=self.language_heads[id2lang_dict[int(language[5])]]
        elif batch_sz==5:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
            head_4=self.language_heads[id2lang_dict[int(language[3])]]
            head_5=self.language_heads[id2lang_dict[int(language[4])]]
        elif batch_sz==4:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
            head_4=self.language_heads[id2lang_dict[int(language[3])]]
        elif batch_sz==3:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
            head_3=self.language_heads[id2lang_dict[int(language[2])]]
        elif batch_sz==2:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]
            head_2=self.language_heads[id2lang_dict[int(language[1])]]
        elif batch_sz==1:
            head_1=self.language_heads[id2lang_dict[int(language[0])]]

        # language_head = self.language_heads[id2lang_dict[int(language)]]
        # id2lang_dict={0: 'EN', 1: 'EL', 2: 'DE', 3: 'TR', 4: 'FR', 5: 'BG', 6: 'HE', 7: 'IT', 8: 'NL'}
        # lang_index = language.item()
        # logits = language_head(outputs.last_hidden_state)

        split_tensors = torch.split(outputs.last_hidden_state, split_size_or_sections=1, dim=0)


        if batch_sz==8:
            logits_1 = head_1(split_tensors[0])                  #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                  #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                  #[:,int(language[2]),:])
            logits_4 = head_4(split_tensors[3])                  #[:,int(language[3]),:])
            logits_5 = head_5(split_tensors[4])                  #[:,int(language[4]),:])
            logits_6 = head_6(split_tensors[5])                  #[:,int(language[5]),:])
            logits_7 = head_7(split_tensors[6])                  #[:,int(language[6]),:])
            logits_8 = head_8(split_tensors[7])                  #[:,int(language[7]),:])
            logits=torch.cat((logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8), dim=0)
        elif batch_sz==7:
            logits_1 = head_1(split_tensors[0])                  #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                  #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                  #[:,int(language[2]),:])
            logits_4 = head_4(split_tensors[3])                  #[:,int(language[3]),:])
            logits_5 = head_5(split_tensors[4])                  #[:,int(language[4]),:])
            logits_6 = head_6(split_tensors[5])                  #[:,int(language[5]),:])
            logits_7 = head_7(split_tensors[6])                  #[:,int(language[6]),:])
            logits=torch.cat((logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7), dim=0)
        elif batch_sz==6:
            logits_1 = head_1(split_tensors[0])                  #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                  #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                  #[:,int(language[2]),:])
            logits_4 = head_4(split_tensors[3])                  #[:,int(language[3]),:])
            logits_5 = head_5(split_tensors[4])                  #[:,int(language[4]),:])
            logits_6 = head_6(split_tensors[5])                  #[:,int(language[5]),:])
            logits=torch.cat((logits_1, logits_2, logits_3, logits_4, logits_5, logits_6), dim=0)
        elif batch_sz==5:
            logits_1 = head_1(split_tensors[0])                  #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                  #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                  #[:,int(language[2]),:])
            logits_4 = head_4(split_tensors[3])                  #[:,int(language[3]),:])
            logits_5 = head_5(split_tensors[4])                  #[:,int(language[4]),:])
            logits=torch.cat((logits_1, logits_2, logits_3, logits_4, logits_5), dim=0)
        elif batch_sz==4:
            logits_1 = head_1(split_tensors[0])                   #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                   #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                   #[:,int(language[2]),:])
            logits_4 = head_4(split_tensors[3])                   #[:,int(language[3]),:])
            logits=torch.cat((logits_1, logits_2, logits_3, logits_4), dim=0)
        elif batch_sz==3:
            logits_1 = head_1(split_tensors[0])                   #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                   #[:,int(language[1]),:])
            logits_3 = head_3(split_tensors[2])                   #[:,int(language[2]),:])
            logits=torch.cat((logits_1, logits_2, logits_3), dim=0)
        elif batch_sz==2:
            logits_1 = head_1(split_tensors[0])                   #[:,int(language[0]),:])
            logits_2 = head_2(split_tensors[1])                   #[:,int(language[1]),:])
            logits=torch.cat((logits_1, logits_2), dim=0)
        elif batch_sz==1:
            logits = head_1(split_tensors[0])                     #[:,int(language[0]),:])

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float)
                loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor.to(labels.device))
            else:
                loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#################################################################################################################################################################
#################################################################################################################################################################

# class CustomExtraHeadRobertaForSequenceClassification(RobertaPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )

#         # self.extra_transformer_layer = RobertaLayer(config)

#         self.extra_transformer_layer_1 = RobertaLayer(config)
#         self.extra_transformer_layer_2 = RobertaLayer(config)
#         self.extra_transformer_layer_3 = RobertaLayer(config)

#         self.pooler = RobertaPooler(config)
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(768, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs.last_hidden_state

#         # sequence_output = self.extra_transformer_layer(sequence_output)

#         sequence_output = self.extra_transformer_layer_1(sequence_output)
#         sequence_output = self.extra_transformer_layer_2(sequence_output[0])
#         sequence_output = self.extra_transformer_layer_3(sequence_output[0])


#         sequence_output = self.pooler(sequence_output[0])

#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#################################################################################################################################################################
#################################################################################################################################################################
