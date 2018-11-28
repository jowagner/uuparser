from bilstm import BiLSTM
from options_manager import OptionsManager
import dynet as dy
import random
import h5py # need to pip install this into environment
import numpy as np
import codecs

class FeatureExtractor(object):
    def __init__(self,model,options,words,rels,langs,w2i,ch,nnvecs):
        self.model = model
        self.disableBilstm = options.disable_bilstm
        self.multiling = options.use_lembed and options.multiling
        self.lstm_output_size = options.lstm_output_size
        self.char_lstm_output_size = options.char_lstm_output_size
        self.word_emb_size = options.word_emb_size
        self.char_emb_size = options.char_emb_size
        self.lang_emb_size = options.lang_emb_size
        self.elmo_emb_size = options.elmo_emb_size
        self.wordsCount = words
        self.vocab = {word: ind+2 for word, ind in w2i.iteritems()} # +2 for MLP padding vector and OOV vector
        self.chars = {char: ind+1 for ind, char in enumerate(ch)} # +1 for OOV vector
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.nnvecs = nnvecs
        if langs:
            self.langs = {lang: ind+1 for ind, lang in enumerate(langs)} # +1 for padding vector
        else:
            self.langs = None
        self.irels = rels
        self.external_embedding = None
        if options.external_embedding is not None:
            self.get_external_embeddings(options.external_embedding)

        self.use_elmo = options.use_elmo
        if self.use_elmo:
            self.elmo_layer = options.elmo_layer
            if self.elmo_layer == "average":
                print "using averaged ELMo representation"

        lstm_input_size = ((self.word_emb_size) +
                           (self.edim if self.external_embedding is not None else 0) +
                           (self.elmo_emb_size if self.use_elmo else 0) +
                           (self.lang_emb_size if self.multiling else 0) +
                           (2 * self.char_lstm_output_size))

        print "LSTM input size: ", lstm_input_size

        if not self.disableBilstm:
            self.bilstm1 = BiLSTM(lstm_input_size, self.lstm_output_size, self.model,
                                  dropout_rate=0.33)
            self.bilstm2 = BiLSTM(2* self.lstm_output_size,
                                  self.lstm_output_size, self.model,
                                  dropout_rate=0.33)
        else:
            self.lstm_output_size = int(lstm_input_size * 0.5)

        self.char_bilstm = BiLSTM(self.char_emb_size,
                                  self.char_lstm_output_size, self.model,
                                  dropout_rate=0.33)

        self.clookup = self.model.add_lookup_parameters((len(ch) + 1, self.char_emb_size))
        self.wlookup = self.model.add_lookup_parameters((len(words) + 2, self.word_emb_size))
        if self.multiling and self.lang_emb_size > 0:
            self.langslookup = self.model.add_lookup_parameters((len(langs) + 1, self.lang_emb_size))

        #used in the PaddingVec
        self.word2lstm = self.model.add_parameters((self.lstm_output_size * 2, lstm_input_size))
        self.word2lstmbias = self.model.add_parameters((self.lstm_output_size *2))
        self.chPadding = self.model.add_parameters((self.char_lstm_output_size *2))

    def Init(self):
        evec = self.elookup[1] if self.external_embedding is not None else None
        paddingWordVec = self.wlookup[1]
        paddingLangVec = self.langslookup[0] if self.multiling and self.lang_emb_size > 0 else None

        self.paddingVec = dy.tanh(self.word2lstm.expr() * dy.concatenate(filter(None,
                                                                                [paddingWordVec,
                                                                                 evec,
                                                                                 self.chPadding.expr(),
                                                                                 paddingLangVec])) + self.word2lstmbias.expr() )
        self.empty = self.paddingVec if self.nnvecs == 1 else dy.concatenate([self.paddingVec for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train, om):
        root_count = 0
        for root in sentence:
            wordcount = float(self.wordsCount.get(root.norm, 0))
            noDropFlag =  not train or (random.random() < (wordcount/(0.25+wordcount)))
            root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if noDropFlag else 0]
            self.get_char_vector(root,train)

            if self.external_embedding is not None:
                if not noDropFlag and random.random() < 0.5:
                    root.evec = self.elookup[0]
                elif root.form in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.norm]]
                else:
                    root.evec = self.elookup[0]
            else:
                root.evec = None

            if self.multiling:
                # TODO: Why do we re-calculate langvec for every word?
                # It usually changes only per document or per sentence.
                # (We may want to set the weights for each token some day
                # though.)
                if om.weighted_tb and om.tb_weights:
                    root.langvec = self.get_weighted_tbemb(entry, om)
                elif om.weighted_tb and om.tb_weights_from_file:
                    root.langvec = self.get_weighted_tbemb(entry, root_entry)
                else:
                    root.langvec = self.langslookup[self.langs[root.language_id]] if self.lang_emb_size > 0 else None
            else:
                root.langvec = None

            if self.use_elmo:
                if self.elmo_layer == "average":
                    if cur_word_index < 0:
                        root.elmo_vec = dy.zeros(1024)
                    else:
                        root.elmo_vec = elmo_embeddings[cur_word_index] # we only use the current word index on valid words and not on the placeholder root token.
            else:
                elmo_vec = None


            root.vec = dy.concatenate(filter(None, [root.wordvec,
                                                        root.evec,
                                                        root.chVec,
                                                        root.langvec])) #TODO: root.elmo_vec
        if not self.disableBilstm:
            self.bilstm1.set_token_vecs(sentence,train)
            self.bilstm2.set_token_vecs(sentence,train)

    def get_char_vector(self,root,train):
        if root.form == "*root*": # no point running a character analysis over this placeholder token
            root.chVec = self.chPadding.expr() # use the padding vector if it's the root token
        else:
            char_vecs = []
            for char in root.form:
                char_vecs.append(self.clookup[self.chars.get(char,0)])
            root.chVec = self.char_bilstm.get_sequence_vector(char_vecs,train)


    def get_external_embeddings(self,external_embedding_file):
        external_embedding_fp = open(external_embedding_file, 'r') # removed utf-8
        external_embedding_fp.readline()
        self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
        external_embedding_fp.close()

        self.edim = len(self.external_embedding.values()[0])
        self.noextrn = [0.0 for _ in xrange(self.edim)] #???
        self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
        self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
        for word, i in self.extrnd.iteritems():
            self.elookup.init_row(i, self.external_embedding[word])
        self.extrnd['*PAD*'] = 1
        self.extrnd['*INITIAL*'] = 2

        print 'Load external embedding. Vector dimensions', self.edim


    def compute_elmo_embeddings(self, data, options, data_type):
        print data_type
        tokenized_sents = []
        elmo_sent_file = os.path.join(options.elmo_output_dir, '%s_concat_sentences.txt') % (data_type)
        for sentence in data:
            tokenized_sent = [entry.form for entry in sentence if isinstance(entry, utils.ConllEntry) and not entry.form == u"*root*"]
            tokenized_sent = [word.encode('utf-8') for word in tokenized_sent]
            tokenized_sents.append(tokenized_sent)

        newline_sents = ('\n'.join(' '.join(map(str, sent)) for sent in tokenized_sents))
        newline_sents = (newline_sents.decode('utf-8', errors='ignore'))
        with codecs.open(elmo_sent_file, 'w', encoding='utf-8') as f:
            f.write(unicode(newline_sents))

        elmo_script = os.path.join(options.elmo_script_dir, 'gen_elmo.sh')
        gen_elmo_command = os.path.join(elmo_script + ' %s/%s_concat_sentences.txt' + ' %s/%s_concat_sentences.hdf5' + ' --average' ) % (options.elmo_output_dir, data_type, options.elmo_output_dir, data_type)
        print "generating elmo vectors: ", gen_elmo_command
        os.system(gen_elmo_command)
        vecs_file = os.path.join(options.elmo_output_dir, '%s_concat_sentences.hdf5') % (data_type)
        h5py_file = h5py.File(vecs_file, 'r')
        print "finished generating elmo representations."
        return h5py_file