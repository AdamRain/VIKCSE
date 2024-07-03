import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertConfig,
)

from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

from tools import *

from icecream import ic

class SimCSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)

        self.embedding = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.projector = MLPLayer(config)

        self.loss_fct = InfoNCE(temp)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        ## 先get input里的[MASK]位置
        mask_index = (input_ids==103).nonzero(as_tuple=True) # #######################
        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            # return representation   #######################
            mask_output0 = representation.last_hidden_state[mask_index[0],mask_index[1]]
            return representation, mask_output0
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.encoder(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = self.projector(representation_0.last_hidden_state[:, 0])
        pooled_output_1 = self.projector(representation_1.last_hidden_state[:, 0])

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class DirectCSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)

        self.embedding = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.projector = MLPLayer(config)
        self.cut_dim = config.cut_dim

        self.loss_fct = InfoNCE(temp)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        ### start here!!
        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            return representation
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.encoder(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        # pooled_output_0 = self.projector(representation_0.last_hidden_state[:, 0])
        # pooled_output_1 = self.projector(representation_1.last_hidden_state[:, 0])
        output_0 = representation_0.last_hidden_state[:, 0][:,:self.cut_dim]
        output_1 = representation_1.last_hidden_state[:, 0][:,:self.cut_dim]

        loss = self.loss_fct(output_0,output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class BYOLSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)
        self.decay = config.decay
        self.online_embedding = BertEmbeddings(config)
        self.online_encoder = BertEncoder(config)
        self.online_projector = MLPLayer(config)
        self.online_predictor = MLPLayer(config)

        self.loss_fct = BYOLMSE(temp)
        
        self.init_weights()

    def prepare(self):
        self.target_embedding = EMA(self.online_embedding, decay = self.decay)
        self.target_encoder = EMA(self.online_encoder,decay = self.decay)
        self.target_projector = EMA(self.online_projector,decay = self.decay)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        ## 先get input里的[MASK]位置
        mask_index = (input_ids==103).nonzero(as_tuple=True) # #######################

        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.online_embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.online_encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            # return representation   #######################
            mask_output0 = representation.last_hidden_state[mask_index[0],mask_index[1]]
            return representation, mask_output0
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.target_embedding.update(self.online_embedding)
        self.target_encoder.update(self.online_encoder)
        self.target_projector.update(self.online_projector)

        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.online_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.target_embedding.model(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.online_encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.target_encoder.model(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = self.online_projector(representation_0.last_hidden_state[:, 0])
        pooled_output_1 = self.target_projector.model(representation_1.last_hidden_state[:, 0])

        pooled_output_0 = self.online_predictor(pooled_output_0)

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class DirectBYOLSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)
        self.decay = config.decay
        self.cut_dim = config.cut_dim
        self.online_embedding = BertEmbeddings(config)
        self.online_encoder = BertEncoder(config)

        self.loss_fct = BYOLMSE(temp)
        
        self.init_weights()

    def prepare(self):
        self.target_embedding = EMA(self.online_embedding, decay = self.decay)
        self.target_encoder = EMA(self.online_encoder, decay = self.decay)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        ## 先get input里的[MASK]位置
        mask_index = (input_ids==103).nonzero(as_tuple=True) # #######################

        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.online_embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.online_encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            # return representation   #######################
            mask_output0 = representation.last_hidden_state[mask_index[0],mask_index[1]]
            return representation, mask_output0
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.target_embedding.update(self.online_embedding)
        self.target_encoder.update(self.online_encoder)

        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.online_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.target_embedding.model(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.online_encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.target_encoder.model(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = representation_0.last_hidden_state[:, 0][:,:self.cut_dim]        
        pooled_output_1 = representation_1.last_hidden_state[:, 0][:,:self.cut_dim]

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class PromptModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)
        self.bert_model = BertModel(config)
        self.mlp = MLPLayer(config)
        self.loss_fct = InfoNCE(temp)
        self.init_weights()
              
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        
        ## 先get input里的[MASK]位置
        mask_index = (input_ids==103).nonzero(as_tuple=True) 
        if sent_emb:
            output0 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
            mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]
            return output0, mask_output0
            
        output0 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        output1 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        
        #返回last_hidden_state里mask位置的embedding
        
        mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]
        mask_output1 = output1.last_hidden_state[mask_index[0],mask_index[1]]
        
        mask_output0 = self.mlp(mask_output0)
        mask_output1 = self.mlp(mask_output1)
        
        loss = self.loss_fct(mask_output0, mask_output1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[mask_output0,mask_output1],
            attentions=None,
        )

class VisualGuidedPromptModel(BertPreTrainedModel):
    def __init__(self, config, visual_pool, text_pool, temp=0.05):
        super().__init__(config)
        self.bert_model = BertModel(config)

        self.mlp = MLPLayer(config)
        # self.mlp = ProjectionMLP(config)

        ##################### Test5\Test6 ####################################
        # self.visual_bert = BertModel(BertConfig(max_position_embeddings=77))
        ######################################################################
        ##################### Test1 ##########################################
        self.trans_linear0 = nn.Sequential(nn.Linear(768, 512), nn.Tanh())
        self.trans_linear1 = nn.Sequential(nn.Linear(768, 512), nn.Tanh())
        ######################################################################
        self.visual_pool = visual_pool

        self.visual_pool_norm = (visual_pool / visual_pool.norm(dim=1, keepdim=True)).transpose(0, 1)

        # self.text_pool = text_pool

        # self.text_pool_norm = (text_pool / text_pool.norm(dim=1, keepdim=True)).transpose(0, 1)
        
        self.loss_fct = InfoNCE(temp)

        # self.loss_fct_w_neg = InfoNCEWithQueue(temp)

        self.temp_denosie = False

        self.n = 3

        self.init_weights()
              
    def get_similar_img_feature_from_img_pool(self, pool, pool_norm, text_feature, n=1):
        # ic(text_feature.shape) # torch.Size([122, 768])
        text_feature_norm = text_feature / text_feature.norm(dim=1, keepdim=True)  # torch.Size([122, 768])
        similarities = torch.matmul(text_feature_norm, pool_norm) # torch.Size([122, 29000])
        # _, max_indices = torch.max(similarities, dim=1)
        max_value, max_indices = torch.sort(similarities, dim=1, descending=True)
        # ic(max_value.shape)  # torch.Size([122, 29000])
        # ic(max_indices.shape)  # torch.Size([122, 29000])
        if n==1:
            closest_vector = pool[max_indices[:,0]]
        else:
            closest_vector = pool[max_indices[:,:n]] # torch.Size([122, 2, 768])
        return closest_vector


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        
        ## 先get input里的[MASK]位置
        # input_ids.shape: torch.Size([batch_size, text_length]) (128, 64)
        mask_index = (input_ids==103).nonzero(as_tuple=True)  # 128-->122 因为有的句子没有mask token，筛选掉了，所以batch_size<128
        
        # ic| input_ids[98]: tensor([  101,  2023,  6251,  1024,  1000,  1031,  1996,  3395,  1997,  1996,
        #                             4169,  2003,  2404,  8082,  2135,  2008,  1997,  3814,  6437,  1997,
        #                             5978,  2652,  1996, 26504,  1010,  2108, 14924,  2098,  2011,  1037,
        #                             2317,  1011, 10681, 18638, 11228, 20051,  3775,  6274,  2010,  2192,
        #                             1010,  1998,  3427,  2058,  2011,  1037,  2417,  1011, 15026,  9684,
        #                             6819,  1997,  3577,  1012,  1033,  1000,  2965,   103,   999,   102,
        #                             0,     0,     0,     0], device='cuda:0')
        #     input_ids[98].shape: torch.Size([64])
        # ic| input_ids[99]: tensor([  101,  2023,  6251,  1024,  1000,  1031,  3833,  1999,  2997,  2005,
        #                             1996, 12951,  5315,  1997, 19031,  9080, 19666,  6776,  2102,  1010,
        #                             1996,  3375,  2001,  2881,  2000, 11865, 10270,  4014,  1996,  3791,
        #                             1997,  1996,  2103,  1005,  1055,  3644,  2313,  1010,  2040,  2018,
        #                             2042,  2302,  1037,  2173,  1997,  7425,  2144,  1996,  4260, 13433,
        #                         16523,  5358,  2043, 28381,  1005,  1055,  2093, 13067,  2015,  2020,
        #                             1033,  1000,  2965,   102], device='cuda:0')   # 第99个句子里没有mask token 103，故删除了该句子
        # 应该是数据处理的时候，有的句子太长了切了一下，导致的刚好切掉的两部分，有一部分没有mask token
        #     input_ids[99].shape: torch.Size([64])
        # ic| input_ids[100]: tensor([  101,  2023,  6251,  1024,  1000,  1031,  2009,  2950,  2195,  6058,
        #                             2642,  9435,  1997,  4940,  2008, 17010,  4600,  2000,  4450,  1010,
        #                             2096,  1996,  2062,  3541,  4664,  4494,  8053,  4600,  2005,  1996,
        #                             13350,  1998, 10342,  1012,  1033,  1000,  2965,   103,   999,   102,
        #                                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                                 0,     0,     0,     0], device='cuda:0')

        # ic(input_ids[:3])  
        # tensor([[  101,  2023,  6251,  1997,  1000,  1996,  3109, 11266,  2034,  2387,
        #                      2895,  2114,  1996,  2887,  2006,  1015,  2244,  3826,  1010,  2043,
        #                      7299,  2125,  2915,  2091,  1037, 10556, 29092,  6182,  1044,  2620,
        #                      2243,  1000,  6253,  1000,  3909,  4049,  1012,  1000,  2965,   103,
        #                      1012,   102,     0,     0,     0,     0,     0,     0,     0,     0,
        #                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                         0,     0,     0,     0],
        #                    [  101,  2023,  6251,  1997,  1000,  1996,  7544,  1005,  1055,  5945,
        #                      2020, 23932,  2455,  1010,  2593,  2312,  2035,  4140,  8163,  2030,
        #                      2235,  2035,  4140,  8163,  2241,  2006,  1996,  4635,  1997,  1996,
        #                      3265,  5945,  2386,  1012,  1000,  2965,   103,  1012,   102,     0,
        #                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                         0,     0,     0,     0],
        #                    [  101,  2023,  6251,  1997,  1000, 10958, 16313,  2102,  2038,  2550,
        #                      1998,  2517,  2774,  2005,  5557, 13250,  5017,  1010, 11382,  2401,
        #                      2527,  1010,  5904,  5623,  1010,  2310, 17625,  1010, 24892,  7082,
        #                      2099,  1010,  4388, 15125,  1010,  6249,  5586, 16237,  1010,  1998,
        #                      2500,  1012,  1000,  2965,   103,  1012,   102,     0,     0,     0,
        #                         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #                         0,     0,     0,     0]], device='cuda:0')
        

        # ic(attention_mask.shape)      # torch.Size([128, 64])
        # ic(token_type_ids.shape)      # torch.Size([128, 64])
        # ic(input_ids.shape)           # torch.Size([128, 64])
        # ic(mask_index[0].shape)       # torch.Size([122])     mask在哪句话
        # ic(mask_index[1].shape)       # torch.Size([122])     mask在那句话的哪个位置
        template_index = []
        if self.temp_denosie:
            template_index_list = []
            for i in range(input_ids.shape[0]):
                idx = torch.zeros(input_ids.shape[1], device='cuda', dtype=torch.int64)
                if i in mask_index[0]:
                    idx[:5] = torch.tensor([101,  2023,  6251,  1997,  1000,], device='cuda', dtype=torch.int64)  # token: This sentence of "
                    if input_ids[i][-1] == 102:
                        idx[mask_index[1][mask_index[0]==i]-2:mask_index[1][mask_index[0]==i]+2] = torch.tensor([1000, 2965, 103, 102], device='cuda', dtype=torch.int64) # token: " means [MASK]
                    else:
                        idx[mask_index[1][mask_index[0]==i]-2:mask_index[1][mask_index[0]==i]+3] = torch.tensor([1000, 2965, 103, 1012, 102], device='cuda', dtype=torch.int64) # token: " means [MASK] !
                    template_index_list.append(idx)
                else:
                    idx[0] = torch.tensor([101], device='cuda', dtype=torch.int64)
                    idx[-1] = torch.tensor([102], device='cuda', dtype=torch.int64)
                    template_index_list.append(idx)
            template_index = torch.stack(template_index_list).cuda()

            

        if sent_emb:
            output0 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
            mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]
            return output0, mask_output0
            
        output0 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        output1 = self.bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        
        #返回last_hidden_state里mask位置的embedding
        
        # Template Denoising
        if self.temp_denosie:
            with torch.no_grad():
                template_output = self.bert_model(input_ids=template_index,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids, 
                                    return_dict=return_dict)
            output0.last_hidden_state -= template_output.last_hidden_state
            output1.last_hidden_state -= template_output.last_hidden_state

        mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output1 = output1.last_hidden_state[mask_index[0],mask_index[1]]  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output0 = self.mlp(mask_output0)  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output1 = self.mlp(mask_output1)
        
        text_loss = self.loss_fct(mask_output0, mask_output1)

        cls_out0 = output0.last_hidden_state[:, 0] #self.mlp(output0.last_hidden_state[:, 0])
        cls_out1 = output1.last_hidden_state[:, 0] #self.mlp(output1.last_hidden_state[:, 0])

        # Test0: 直接对比文本中间变量CLS和MASK token的特征
        # mask_loss = self.loss_fct(output0.last_hidden_state[:, 0], output1.last_hidden_state[:, 0])
        
        # Test1: 获得从image feature pool中获得的最相似embedding，由于feature是512x512，需要embedding映射
        vis_mask_output0 = self.trans_linear0(mask_output0)  # 适配 CLIP visual的维度 512
        vis_mask_output1 = self.trans_linear0(mask_output1)  # 适配 CLIP visual的维度 512

        vis_cls_output0 = self.trans_linear1(cls_out0)   # cls token   output0.last_hidden_state[:, 0]
        vis_cls_output1 = self.trans_linear1(cls_out1)   # cls token   output1.last_hidden_state[:, 0]

        with torch.no_grad():
            vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, vis_mask_output0)
            vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, vis_mask_output1)
            vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, vis_cls_output0)
            vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, vis_cls_output1)
        
        vis_mask_loss  = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        vis_cls_loss = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)

        # Test2: 直接从image feature pool中获得的最相似embedding，feature是768x768
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, mask_output0)
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, mask_output1)
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, cls_out0)
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, cls_out1)

        # # 计算图像级别对比loss
        # vis_mask_loss  = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        # vis_cls_loss = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)

        # Test3: 拼接特征, 选取topn个最相似特征，其中，后四个作为负样本，构建带有硬负样本的对比loss
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, mask_output0, n=self.n)  # torch.Size([batch_size, n, 768])
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, mask_output1, n=self.n)  # torch.Size([batch_size, n, 768])
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, output0.last_hidden_state[:, 0], n=self.n)  # torch.Size([batch_size, n, 768])
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(self.visual_pool, self.visual_pool_norm, output1.last_hidden_state[:, 0], n=self.n)  # torch.Size([batch_size, n, 768])


        # neg_cated_vis_mask_img_feats0 = torch.cat((vis_mask_img_feats0[:,1:], vis_mask_img_feats1[:,1:]),dim=1) # torch.Size([batch_size, 2*(n-1), 768])
        # neg_cated_vis_cls_img_feats0 = torch.cat((vis_cls_img_feats0[:,1:], vis_cls_img_feats1[:,1:]),dim=1)    # torch.Size([batch_size, 2*(n-1), 768])

        # vis_mask_loss_w_neg = self.loss_fct_w_neg(vis_mask_img_feats0[:,0], vis_mask_img_feats1[:,0], neg_cated_vis_mask_img_feats0)
        # vis_cls_loss_w_neg = self.loss_fct_w_neg(vis_cls_img_feats0[:,0], vis_cls_img_feats1[:,0], neg_cated_vis_cls_img_feats0)
        

        # Test4: 直接从text feature pool中获得的最相似embedding，feature是768x768, 对比text feature
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(mask_output0)
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(mask_output1)
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(output0.last_hidden_state[:, 0])
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(output1.last_hidden_state[:, 0])

        # # 计算图像对比loss
        # vis_mask_loss  = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        # vis_cls_loss = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)


        # Test5: 原句token过一下预训练的多模态文本BERT，然后拿到对应位置的 MASK embedding 和 CLS embedding， 然后直接对比
        # with torch.no_grad():
        #     output2 = self.visual_bert(input_ids=input_ids,
        #                             attention_mask=attention_mask,
        #                             token_type_ids=token_type_ids, 
        #                             return_dict=return_dict)
        # mask_output2 = output2.last_hidden_state[mask_index[0],mask_index[1]]   # torch.Size([122, 768])
        # text_loss0 = self.loss_fct(mask_output0, mask_output2)
        # text_loss1 = self.loss_fct(mask_output1, mask_output2)


        # Test6: 原句token过一下预训练的多模态文本BERT，然后拿到对应位置的 MASK embedding 和 CLS embedding， 然后从image feature pool中找到相似的图像进行对比
        # with torch.no_grad():
        #     output2 = self.visual_bert(input_ids=input_ids,
        #                             attention_mask=attention_mask,
        #                             token_type_ids=token_type_ids, 
        #                             return_dict=return_dict)
        # mask_output2 = output2.last_hidden_state[mask_index[0],mask_index[1]]   # torch.Size([122, 768])
        
        
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(mask_output0)
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(mask_output1)
        #     vis_mask_img_feats2 = self.get_similar_img_feature_from_img_pool(mask_output2)
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(output0.last_hidden_state[:, 0])
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(output1.last_hidden_state[:, 0])
        #     vis_cls_img_feats2 = self.get_similar_img_feature_from_img_pool(output2.last_hidden_state[:, 0])

        # # 计算图像对比loss
        # vis_mask_loss_0_1 = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        # vis_mask_loss_0_2 = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats2)
        # vis_mask_loss_1_2 = self.loss_fct(vis_mask_img_feats1, vis_mask_img_feats2)

        # vis_cls_loss_0_1 = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)
        # vis_cls_loss_0_2 = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats2)
        # vis_cls_loss_1_2 = self.loss_fct(vis_cls_img_feats1, vis_cls_img_feats2)


        # Test7：将图像数据集进行裁切，扩展，然后再实验Test2/Test3
        # Test7-1: 扩展1倍


        # Test7-2: 扩展2倍


        # Test7-3: 随机删除一半数据



        # Test8: 第一步中的微调是否有必要？直接冻结预训练模型只训练projection层，然后获得pure pretrained model的feature pool



        # 计算总loss

        alpha = 0.5
        beta = 0.5

        # original:
        # loss = text_loss
        # Test0:
        # loss = text_loss + mask_loss
        # loss = alpha*text_loss+beta*text_loss1

        # Test1/Test2/Test4:
        # loss = text_loss + alpha*vis_mask_loss + beta*vis_cls_loss
        
        loss = text_loss + alpha*vis_mask_loss / torch.abs(vis_mask_loss - text_loss).detach() + beta*vis_cls_loss / torch.abs(vis_cls_loss - text_loss).detach()
        
        # Test3:
        # loss = text_loss + alpha*vis_mask_loss_w_neg / torch.abs(vis_mask_loss_w_neg - text_loss).detach() + beta*vis_cls_loss_w_neg / torch.abs(vis_cls_loss_w_neg - text_loss).detach()

        
        # Test5:
        # loss = text_loss + 0.5*text_loss0 + 0.5*text_loss1
        # Test6:
        # loss = text_loss + 0.16*vis_mask_loss_0_1 + 0.16*vis_cls_loss_0_1 + 0.16*vis_mask_loss_0_2 + 0.16*vis_cls_loss_0_2 + 0.16*vis_mask_loss_1_2 + 0.16*vis_cls_loss_1_2
        
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[mask_output0,mask_output1], # type: ignore
            attentions=None,
        )


class VisualGuidedPromptModel_roberta(RobertaPreTrainedModel):
    def __init__(self, config, visual_pool,temp=0.05):
        super().__init__(config)
        self.roberta_model = RobertaModel(config)

        self.mlp = MLPLayer(config)
        # self.mlp = ProjectionMLP(config)

        ##################### Test5\Test6 ####################################
        # self.visual_bert = BertModel(BertConfig(max_position_embeddings=77))
        ######################################################################
        ##################### Test1 ##########################################
        self.trans_linear0 = nn.Sequential(nn.Linear(768, 512), nn.Tanh())
        self.trans_linear1 = nn.Sequential(nn.Linear(768, 512), nn.Tanh())
        ######################################################################
        
        self.visual_pool = visual_pool

        self.loss_fct = InfoNCE(temp)
        
        self.temp_denosie = False

        self.loss_fct_w_neg = InfoNCEWithQueue(temp)

        self.init_weights()
              
    def get_similar_img_feature_from_img_pool(self, text_feature, n=1):
        # ic(text_feature.shape) # torch.Size([122, 768])
        text_feature_norm = text_feature / text_feature.norm(dim=1, keepdim=True)  # torch.Size([122, 768])
        visual_pool_norm = self.visual_pool / self.visual_pool.norm(dim=1, keepdim=True)   # torch.Size([29000, 768])
        similarities = torch.matmul(text_feature_norm, visual_pool_norm.transpose(0, 1)) # torch.Size([122, 29000])
        # _, max_indices = torch.max(similarities, dim=1)
        max_value, max_indices = torch.sort(similarities, dim=1, descending=True)
        if n==1:
            closest_vector = self.visual_pool[max_indices[:,0]]
        else:
            closest_vector = self.visual_pool[max_indices[:,:n]] # torch.Size([122, 2, 768])
        return closest_vector


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        
        ## 先get input里的[MASK]位置
        # input_ids.shape: torch.Size([batch_size, text_length]) (128, 64)
        mask_index = (input_ids==50264).nonzero(as_tuple=True)  # 128-->122 因为有的句子没有mask token，筛选掉了，所以batch_size<128

        # <s> The  sentence  :     \'   [S] \'  means <mask> !      </s>
        #  0  133  3645      4832  128      128 839   50264  27785  2
        # <pad>: 1
        template_index = []
        if self.temp_denosie:
            template_index_list = []
            for i in range(input_ids.shape[0]):
                idx = torch.ones(input_ids.shape[1], device='cuda', dtype=torch.int64)
                if i in mask_index[0]:
                    idx[:5] = torch.tensor([0,  133,  3645,  4832,  128], device='cuda', dtype=torch.int64)  # token: <s> The sentence : \'
                    if input_ids[i][-1] == 2:
                        idx[mask_index[1][mask_index[0]==i]-3:mask_index[1][mask_index[0]==i]+1] = torch.tensor([128, 839, 50264, 2], device='cuda', dtype=torch.int64) # token:\' means <mask>
                    else:
                        idx[mask_index[1][mask_index[0]==i]-3:mask_index[1][mask_index[0]==i]+2] = torch.tensor([128, 839, 50264, 27785, 2], device='cuda', dtype=torch.int64) # token: \' means <mask> !
                    template_index_list.append(idx)
                else:
                    idx[0] = torch.tensor([0], device='cuda', dtype=torch.int64)
                    idx[-1] = torch.tensor([2], device='cuda', dtype=torch.int64)
                    template_index_list.append(idx)
            template_index = torch.stack(template_index_list).cuda()

        if sent_emb:
            output0 = self.roberta_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
            mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]
            return output0, mask_output0
            
        output0 = self.roberta_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        output1 = self.roberta_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        
        # Template Denoising
        if self.temp_denosie:
            with torch.no_grad():
                template_output = self.roberta_model(input_ids=template_index,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids, 
                                        return_dict=return_dict)
            output0.last_hidden_state -= template_output.last_hidden_state
            output1.last_hidden_state -= template_output.last_hidden_state


        #返回last_hidden_state里mask位置的embedding
        
        mask_output0 = output0.last_hidden_state[mask_index[0],mask_index[1]]  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output1 = output1.last_hidden_state[mask_index[0],mask_index[1]]  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output0 = self.mlp(mask_output0)  # mask_output0.shape: torch.Size([batch_size, 768])
        mask_output1 = self.mlp(mask_output1)
        
        text_loss = self.loss_fct(mask_output0, mask_output1)

        # Test0: 直接对比文本中间变量CLS和MASK token的特征
        # mask_loss = self.loss_fct(output0.last_hidden_state[:, 0], output1.last_hidden_state[:, 0])

        
        # Test1: 获得从image feature pool中获得的最相似embedding，由于feature是512x512，需要embedding映射
        vis_mask_output0 = self.trans_linear0(mask_output0)  # 适配 CLIP visual的维度 512
        vis_mask_output1 = self.trans_linear0(mask_output1)  # 适配 CLIP visual的维度 512

        vis_cls_output0 = self.trans_linear1(output0.last_hidden_state[:, 0])   # cls token
        vis_cls_output1 = self.trans_linear1(output1.last_hidden_state[:, 0])

        with torch.no_grad():
            vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(vis_mask_output0)
            vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(vis_mask_output1)
            vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(vis_cls_output0)
            vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(vis_cls_output1)
        
        vis_mask_loss  = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        vis_cls_loss = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)

        # Test2: 直接从image feature pool中获得的最相似embedding，feature是768x768
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(mask_output0)
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(mask_output1)
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(output0.last_hidden_state[:, 0])
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(output1.last_hidden_state[:, 0])

        # # 计算图像对比loss
        # vis_mask_loss  = self.loss_fct(vis_mask_img_feats0, vis_mask_img_feats1)
        # vis_cls_loss = self.loss_fct(vis_cls_img_feats0, vis_cls_img_feats1)


        # Test3: 拼接特征, 选取topn个最相似特征，其中，后四个作为负样本，构建带有硬负样本的对比loss
        # with torch.no_grad():
        #     vis_mask_img_feats0 = self.get_similar_img_feature_from_img_pool(mask_output0, n=5)  # torch.Size([batch_size, n, 768])
        #     vis_mask_img_feats1 = self.get_similar_img_feature_from_img_pool(mask_output1, n=5)  # torch.Size([batch_size, n, 768])
        #     vis_cls_img_feats0 = self.get_similar_img_feature_from_img_pool(output0.last_hidden_state[:, 0], n=5)  # torch.Size([batch_size, n, 768])
        #     vis_cls_img_feats1 = self.get_similar_img_feature_from_img_pool(output1.last_hidden_state[:, 0], n=5)  # torch.Size([batch_size, n, 768])


        # neg_cated_vis_mask_img_feats0 = torch.cat((vis_mask_img_feats0[:,1:], vis_mask_img_feats1[:,1:]),dim=1) # torch.Size([batch_size, 2*(n-1), 768])
        # neg_cated_vis_cls_img_feats0 = torch.cat((vis_cls_img_feats0[:,1:], vis_cls_img_feats1[:,1:]),dim=1)    # torch.Size([batch_size, 2*(n-1), 768])

        # vis_mask_loss_w_neg = self.loss_fct_w_neg(vis_mask_img_feats0[:,0], vis_mask_img_feats1[:,0], neg_cated_vis_mask_img_feats0)
        # vis_cls_loss_w_neg = self.loss_fct_w_neg(vis_cls_img_feats0[:,0], vis_cls_img_feats1[:,0], neg_cated_vis_cls_img_feats0)
        
  
        # 计算总loss

        alpha = 0.5
        beta = 0.5

        # Test1/Test2/Test4:
        loss = text_loss + alpha*vis_mask_loss / torch.abs(vis_mask_loss - text_loss).detach() + beta*vis_cls_loss / torch.abs(vis_cls_loss - text_loss).detach()
        
        # Test3:
        # loss = text_loss + alpha*vis_mask_loss_w_neg/ torch.abs(vis_mask_loss_w_neg - text_loss).detach() + beta*vis_cls_loss_w_neg / torch.abs(vis_cls_loss_w_neg - text_loss).detach()

        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[mask_output0,mask_output1], # type: ignore
            attentions=None,
        )
    


if __name__ == '__main__':
    config = BertConfig.from_pretrained('bert-base-uncased', cache_dir='./cache')
    model = SimCSEModel(config)
    print(model)