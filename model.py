import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, AutoModel, AutoModelForQuestionAnswering
import torch


class BertForQuestionAnswering(nn.Module):
    def __init__(self, model_type: str):
        super(BertForQuestionAnswering, self).__init__()
        if 'deberta' in model_type:
            self.bert = AutoModel.from_pretrained(model_type)
        elif 'bert-' in model_type:
            self.bert = BertModel.from_pretrained(model_type)
        else:
            raise ValueError('Model type!')

        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # [N, L, H] => [N, L, 2]
        self.classifier_coeff = 10
        self.entropy_penalty = 0
        # print(self.classifier_coeff)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch, classifier=False, return_prob=False, **kwargs):
        '''
        each batch is a list of 5 items (training) or 3 items (inference)
            - input_ids: token id of the input sequence
            - attention_mask: mask of the sequence (1 for present, 0 for blank)
            - token_type_ids: indicator of type of sequence.
            -      e.g. in QA, whether it is question or document
            - (training) start_positions: list of start positions of the span
            - (training) end_positions: list of end positions of the span
        '''

        input_ids, attention_masks, token_type_ids = batch[:3]
        # pooler_output, last_hidden_state
        output = self.bert(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_masks)
        sequence_output = output.last_hidden_state
        logits = self.qa_outputs(sequence_output)  # (bs, max_input_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_input_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_input_len)

        if len(batch) == 5:
            start_positions, end_positions = batch[3:]
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            
            if classifier:
                answerable_mask = (start_positions != 0) | (end_positions != 0)
                loss_fct = CrossEntropyLoss()
                
                answerable_logits = self.classification(sequence_output)[:, 0]
                classifier_loss = loss_fct(answerable_logits, answerable_mask.long())
                total_loss += self.classifier_coeff * classifier_loss
                answerable_prob = torch.softmax(answerable_logits, dim=-1)

                total_loss += self.entropy_penalty * (-torch.mean(torch.sum(-answerable_prob * torch.log(answerable_prob), dim=-1)))
                return total_loss, torch.softmax(self.classification(sequence_output[:, 0]), dim=-1)
            return total_loss, None

        elif len(batch) == 3 and not classifier:
            if not return_prob:
                return start_logits, end_logits, None
            else:
                return self.softmax(start_logits), self.softmax(end_logits), None
        elif len(batch) == 3 and classifier:
            if return_prob:
                return self.softmax(start_logits), self.softmax(
                    end_logits), self.softmax(self.classification(sequence_output[:, 0]))
            else:
                return start_logits, end_logits, self.classification(sequence_output[:, 0])
        else:
            raise NotImplementedError()


class BertForQuestionAnsweringSequence(BertForQuestionAnswering):
    def __init__(self, model_type: str):
        super(BertForQuestionAnsweringSequence, self).__init__(model_type=model_type)
        self.classification = nn.Linear(self.bert.config.hidden_size, 2)  # [N, L, H] => [N, L, 2]





class DebertaSQuAD2(nn.Module):
    def __init__(self, model_type: str):
        super(DebertaSQuAD2, self).__init__()
        if model_type == 'deepset/deberta-v3-base-squad2':
            self.bert = AutoModelForQuestionAnswering.from_pretrained(model_type)
        else:
            raise ValueError('Model type!')

    def forward(self, batch, return_prob=False, **kwargs):
        '''
        each batch is a list of 5 items (training) or 3 items (inference)
            - input_ids: token id of the input sequence
            - attention_mask: mask of the sequence (1 for present, 0 for blank)
            - token_type_ids: indicator of type of sequence.
            -      e.g. in QA, whether it is question or document
            - (training) start_positions: list of start positions of the span
            - (training) end_positions: list of end positions of the span
        '''

        input_ids, attention_masks, token_type_ids = batch[:3]
        # pooler_output, last_hidden_state
        output = self.bert(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_masks)

        start_logits, end_logits = output.start_logits, output.end_logits

        if len(batch) == 3:
            if not return_prob:
                return start_logits, end_logits
            else:
                return torch.softmax(start_logits, dim=-1), torch.softmax(end_logits, dim=-1)

        else:
            raise NotImplementedError()
