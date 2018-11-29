from utils import ParseForest, read_conll, write_conll
from multilayer_perceptron import MLP
from feature_extractor import FeatureExtractor
from options_manager import OptionsManager
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np
from copy import deepcopy
import csv
import os
import h5py


class ArcHybridLSTM:
    def __init__(self, words, pos, rels, cpos, langs, w2i, ch, options):
        import dynet as dy # import here so we don't load Dynet if just running parser.py --help for example
        global dy
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model, alpha=options.learning_rate)
        self.activations = {'tanh': dy.tanh, 'sigmoid': dy.logistic, 'relu':
                            dy.rectify, 'tanh3': (lambda x:
                            dy.tanh(dy.cwise_multiply(dy.cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]
        self.oracle = options.oracle
        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.k
        self.use_elmo = options.use_elmo
        self.elmo_layer = options.elmo_layer

        #dimensions depending on extended features
        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)
        self.feature_extractor = FeatureExtractor(self.model,options,words,rels,langs,w2i,ch,self.nnvecs)

        self.irels = self.feature_extractor.irels

        if langs: # define langs here as well for tbidmetadata
            self.langs = {lang: ind+1 for ind, lang in enumerate(langs)} # +1 for padding vector
        else:
            self.langs = None

        mlp_in_dims = options.lstm_output_size*2*self.nnvecs*(self.k+1)
        self.unlabeled_MLP = MLP(self.model, 'unlabeled', mlp_in_dims, options.mlp_hidden_dims,
                                 options.mlp_hidden2_dims, 4, self.activation)
        self.labeled_MLP = MLP(self.model, 'labeled' ,mlp_in_dims, options.mlp_hidden_dims,
                               options.mlp_hidden2_dims,2*len(self.irels)+2,self.activation)


    def __evaluate(self, stack, buf, train):
        """
        ret = [left arc,
               right arc
               shift]

        RET[i] = (rel, transition, score1, score2) for shift, l_arc and r_arc
         shift = 2 (==> rel=None) ; l_arc = 0; r_acr = 1

        ret[i][j][2] ~= ret[i][j][3] except the latter is a dynet
        expression used in the loss, the first is used in rest of training
        """

        #feature rep
        empty = self.feature_extractor.empty
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [empty] for i in xrange(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [empty] for i in xrange(1) ]

        input = dy.concatenate(list(chain(*(topStack + topBuffer))))
        output = self.unlabeled_MLP(input)
        routput = self.labeled_MLP(input)

        #scores, unlabeled scores
        scrs, uscrs = routput.value(), output.value()

        #transition conditions
        left_arc_conditions = len(stack) > 0
        right_arc_conditions = len(stack) > 1
        shift_conditions = buf.roots[0].id != 0
        swap_conditions = len(stack) > 0 and stack.roots[-1].id < buf.roots[0].id

        if not train:
            #(avoiding the multiple roots problem: disallow left-arc from root
            #if stack has more than one element
            left_arc_conditions = left_arc_conditions and not (buf.roots[0].id == 0 and len(stack) > 1)

        uscrs0 = uscrs[0] #shift
        uscrs1 = uscrs[1] #swap
        uscrs2 = uscrs[2] #left-arc
        uscrs3 = uscrs[3] #right-arc

        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            output3 = output[3]


            ret = [ [ (rel, 0, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if left_arc_conditions else [],
                   [ (rel, 1, scrs[3 + j * 2] + uscrs3, routput[3 + j * 2 ] + output3) for j, rel in enumerate(self.irels) ] if right_arc_conditions else [],
                   [ (None, 2, scrs[0] + uscrs0, routput[0] + output0) ] if shift_conditions else [] ,
                    [ (None, 3, scrs[1] + uscrs1, routput[1] + output1) ] if swap_conditions else [] ]
        else:
            s1,r1 = max(zip(scrs[2::2],self.irels))
            s2,r2 = max(zip(scrs[3::2],self.irels))
            s1 += uscrs2
            s2 += uscrs3
            ret = [ [ (r1, 0, s1) ] if left_arc_conditions else [],
                   [ (r2, 1, s2) ] if right_arc_conditions else [],
                   [ (None, 2, scrs[0] + uscrs0) ] if shift_conditions else [] ,
                    [ (None, 3, scrs[1] + uscrs1) ] if swap_conditions else [] ]
        return ret


    def Save(self, filename):
        print 'Saving model to ' + filename
        self.model.save(filename)


    def Load(self, filename):
        print 'Loading model from ' + filename
        self.model.populate(filename)


    def apply_transition(self,best,stack,buf,hoffset):
        if best[1] == 2:
            #SHIFT
            stack.roots.append(buf.roots[0])
            del buf.roots[0]

        elif best[1] == 3:
            #SWAP
            child = stack.roots.pop()
            buf.roots.insert(1,child)

        elif best[1] == 0:
            #LEFT-ARC
            child = stack.roots.pop()
            parent = buf.roots[0]

            #predict rel and label
            child.pred_parent_id = parent.id
            child.pred_relation = best[0]

        elif best[1] == 1:
            #RIGHT-ARC
            child = stack.roots.pop()
            parent = stack.roots[-1]

            child.pred_parent_id = parent.id
            child.pred_relation = best[0]

        #update the representation of head for attaching transitions
        if best[1] == 0 or best[1] == 1:
            #linear order #not really - more like the deepest
            if self.rlMostFlag:
                parent.lstms[best[1] + hoffset] = child.lstms[best[1] + hoffset]
                #actual children
            if self.rlFlag:
                parent.lstms[best[1] + hoffset] = child.vec


    def calculate_cost(self,scores,s0,s1,b,beta,stack_ids):
        if len(scores[0]) == 0:
            left_cost = 1
        else:
            left_cost = len(s0[0].rdeps) + int(s0[0].parent_id != b[0].id and s0[0].id in s0[0].parent_entry.rdeps)


        if len(scores[1]) == 0:
            right_cost = 1
        else:
            right_cost = len(s0[0].rdeps) + int(s0[0].parent_id != s1[0].id and s0[0].id in s0[0].parent_entry.rdeps)


        if len(scores[2]) == 0:
            shift_cost = 1
            shift_case = 0
        elif len([item for item in beta if item.projective_order < b[0].projective_order and item.id > b[0].id ])> 0:
            shift_cost = 0
            shift_case = 1
        else:
            shift_cost = len([d for d in b[0].rdeps if d in stack_ids]) + int(len(s0)>0 and b[0].parent_id in stack_ids[:-1] and b[0].id in b[0].parent_entry.rdeps)
            shift_case = 2


        if len(scores[3]) == 0 :
            swap_cost = 1
        elif s0[0].projective_order > b[0].projective_order:
            swap_cost = 0
            #disable all the others
            left_cost = right_cost = shift_cost = 1
        else:
            swap_cost = 1

        costs = (left_cost, right_cost, shift_cost, swap_cost,1)
        return costs,shift_case


    def get_tbidmetadata(self, options, om, langs):
        # NOTE:
        # similar function to "get_weighted_tbemb" but just returning the tbidmetadata dict and not doing any computation.
        # added this function into ArcHybridLSTM so we can retrieve "tbidmetadata" in parser.py from parser.tbidmetadata where parser is the  ArcHybridLSTM class.
        # overkill? can we not just access the dict from FeatureExtractor?
        # does this achieve the same results?

        tbid2weights = {} # 1 mapping to weight specified on command line
        tbid2tb = {}      # 2 mapping to tbname
        tb2key = {}       # 3 mapping from tbname to index in langslookup
        self.tbidmetadata = {}
        tbidmetadata = self.tbidmetadata

        # 1: Mapping from tbid to weight
        for tb_weight in om.tb_weights.split():
            tbid, weight = tb_weight.split(':')
            if tbid not in tbid2weights:
                tbid2weights[tbid] = weight
            else:
                raise ValueError, 'weight for %r specified more than once' %tbid

        # 2: Mapping from tbid to treebank name
        #    (completeness will be checked at start of step 4)
        with open("../config/tbnames.tsv") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                tb = row[0]
                tbid = row[1]
                if tbid in tbid2weights:
                    tbid2tb[tbid] = tb

        # 3: Mapping from treebank to lang number
        for tb, id_number in self.langs.items():
            tb = str(tb) # deal with mismatch of str and unicode types
            if tb in str(tbid2tb):
                tb2key[tb] = id_number

        # 4: Append all of the above dictionary values based on tbid key
        for tbid, weight in tbid2weights.items():
            if tbid in tbid2tb:
                tbidmetadata[tbid] = []
            else:
                raise ValueError, 'no tbname configured for tbid %r' %tbid

        for tbid, v in tbidmetadata.items():
            tbname = tbid2tb[tbid]
            index  = tb2key[tbname]
            weight = tbid2weights[tbid]
            v.append(tbname)
            v.append(weight)
            v.append(index)
        return tbidmetadata


    def Predict(self, data, om, options):
        pred_index = 0

        if self.use_elmo:
            vecs_file = os.path.join(options.elmo_output_dir, 'en_lines_dev_sentences.hdf5') # sample filename for the moment
            if os.path.exists(vecs_file):
                print("using elmo file {}".format(vecs_file))
                h5py_file = h5py.File(vecs_file, 'r')
            else:
                print 'cannot find elmo file'

        reached_max_swap = 0
        for iSentence, osentence in enumerate(data,1):
            sentence = deepcopy(osentence)
            reached_swap_for_i_sentence = False
            max_swap = 2*len(sentence)
            iSwap = 0

            self.feature_extractor.Init()
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            tokenized_sent = [entry.form for entry in sentence if isinstance(entry, utils.ConllEntry) and not entry.form == u"*root*"]

            if self.use_elmo:
                if self.elmo_layer == "average":
                    elmo_embeddings_np = h5py_file[str(pred_index)][:]
                    assert elmo_embeddings_np.shape[0] == len(tokenized_sent)
                    elmo_embeddings = dy.inputTensor(elmo_embeddings_np)

            self.feature_extractor.getWordEmbeddings(conll_sentence, False, om, elmo_embeddings if self.use_elmo else None)
            stack = ParseForest([])
            buf = ParseForest(conll_sentence)

            hoffset = 1 if self.headFlag else 0

            for root in conll_sentence:
                root.lstms = [root.vec] if self.headFlag else []
                root.lstms += [self.feature_extractor.paddingVec for _ in range(self.nnvecs - hoffset)]
                root.relation = root.relation if root.relation in self.irels else 'runk'


            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.__evaluate(stack, buf, False)
                best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
                if iSwap == max_swap and not reached_swap_for_i_sentence:
                    reached_max_swap += 1
                    reached_swap_for_i_sentence = True
                    print "reached max swap in %d out of %d sentences"%(reached_max_swap, iSentence)
                self.apply_transition(best,stack,buf,hoffset)
                if best[1] == 3:
                    iSwap += 1


            dy.renew_cg()

            #keep in memory the information we need, not all the vectors
            oconll_sentence = [entry for entry in osentence if isinstance(entry, utils.ConllEntry)]
            oconll_sentence = oconll_sentence[1:] + [oconll_sentence[0]]
            for tok_o, tok in zip(oconll_sentence, conll_sentence):
                tok_o.pred_relation = tok.pred_relation
                tok_o.pred_parent_id = tok.pred_parent_id

            pred_index += 1
            yield osentence


    def Train(self, trainData, om, options):
        mloss = 0.0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ninf = -float('inf')
        train_idx = 0

        beg = time.time()
        start = time.time()

        errs = []

        self.feature_extractor.Init()

        if self.use_elmo:
            vecs_file = os.path.join(options.elmo_output_dir, 'en_lines_train_sentences.hdf5') # sample filename for the moment
            if os.path.exists(vecs_file):
                print("using elmo file {}".format(vecs_file))
                h5py_file = h5py.File(vecs_file, 'r')
            else:
                print 'cannot find elmo file' # we are assuming this is done beforehand.

        permutation = list(range(len(trainData)))
        random.shuffle(permutation)

        sentence_dict = {} # store sentences here and access them with permutation key (which is shuffled)
        for iSentence, sentence in enumerate(trainData):
            if iSentence not in sentence_dict.keys():
                sentence_dict[iSentence] = sentence

        print "Length of training data: ", len(trainData)

        for iPermutation in permutation:
            train_idx +=1
            sentence = sentence_dict.get(iPermutation)
            if train_idx % 100 == 0:
                loss_message = 'Processing sentence number: %d'%train_idx + \
                ' Loss: %.3f'%(eloss / etotal)+ \
                ' Errors: %.3f'%((float(eerrors)) / etotal)+\
                ' Labeled Errors: %.3f'%(float(lerrors) / etotal)+\
                ' Time: %.2gs'%(time.time()-start)
                print loss_message
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                lerrors = 0

            sentence = deepcopy(sentence) # ensures we are working with a clean copy of sentence and allows memory to be recycled each time round the loop

            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            tokenized_sent = [entry.form for entry in sentence if isinstance(entry, utils.ConllEntry) and not entry.form == u"*root*"]

            if self.use_elmo:
                if self.elmo_layer == "average":
                    elmo_embeddings_np = h5py_file[str(iPermutation)][:]
                    assert elmo_embeddings_np.shape[0] == len(tokenized_sent)
                    elmo_embeddings = dy.inputTensor(elmo_embeddings_np)

            self.feature_extractor.getWordEmbeddings(conll_sentence, True, om, elmo_embeddings if self.use_elmo else None)

            stack = ParseForest([])
            buf = ParseForest(conll_sentence)
            hoffset = 1 if self.headFlag else 0
            
            
            #train_idx +=1
            for root in conll_sentence:
                root.lstms = [root.vec] if self.headFlag else []
                root.lstms += [self.feature_extractor.paddingVec for _ in range(self.nnvecs - hoffset)]
                root.relation = root.relation if root.relation in self.irels else 'runk'


            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.__evaluate(stack, buf, True)

                #to ensure that we have at least one wrong operation
                scores.append([(None, 4, ninf ,None)])

                stack_ids = [sitem.id for sitem in stack.roots]

                s1 = [stack.roots[-2]] if len(stack) > 1 else []
                s0 = [stack.roots[-1]] if len(stack) > 0 else []
                b = [buf.roots[0]] if len(buf) > 0 else []
                beta = buf.roots[1:] if len(buf) > 1 else []

                costs, shift_case = self.calculate_cost(scores,s0,s1,b,beta,stack_ids)

                bestValid = list(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or s[1] == 3 or  s[0] == s0[0].relation ) ))
                if len(bestValid) <1:
                    print "===============dropping a sentence==============="
                    break

                bestValid = max(bestValid, key=itemgetter(2))
                bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or ( s[1] != 2 and s[1] != 3 and s[0] != s0[0].relation ) ), key=itemgetter(2))

                #force swap
                if costs[3]== 0:
                    best = bestValid
                else:
                    #select a transition to follow
                    # + aggresive exploration
                    #1: might want to experiment with that parameter
                    if bestWrong[1] == 3:
                        best = bestValid
                    else:
                        best = bestValid if ( (not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

                #updates for the dynamic oracle
                if best[1] == 2:
                    #SHIFT
                    if shift_case ==2:
                        if b[0].parent_entry.id in stack_ids[:-1] and b[0].id in b[0].parent_entry.rdeps:
                            b[0].parent_entry.rdeps.remove(b[0].id)
                        blocked_deps = [d for d in b[0].rdeps if d in stack_ids]
                        for d in blocked_deps:
                            b[0].rdeps.remove(d)

                elif best[1] == 0 or best[1] == 1:
                    #LA or RA
                    child = s0[0]
                    s0[0].rdeps = []
                    if s0[0].id in s0[0].parent_entry.rdeps:
                        s0[0].parent_entry.rdeps.remove(s0[0].id)

                self.apply_transition(best,stack,buf,hoffset)

                if bestValid[2] < bestWrong[2] + 1.0:
                    loss = bestWrong[3] - bestValid[3]
                    mloss += 1.0 + bestWrong[2] - bestValid[2]
                    eloss += 1.0 + bestWrong[2] - bestValid[2]
                    errs.append(loss)

                #labeled errors
                if best[1] != 2 and best[1] !=3 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                    lerrors += 1
                    #attachment error
                    if child.pred_parent_id != child.parent_id:
                        eerrors += 1

                if best[1] == 0 or best[1] == 2:
                    etotal += 1

            #footnote 8 in Eli's original paper
            if len(errs) > 50: # or True:
                eerrs = dy.esum(errs)
                scalar_loss = eerrs.scalar_value() #forward
                eerrs.backward()
                self.trainer.update()
                errs = []
                lerrs = []

                dy.renew_cg()
                self.feature_extractor.Init()



        if len(errs) > 0:
            eerrs = (dy.esum(errs))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            dy.renew_cg()

        self.trainer.update()
        print "Loss: ", mloss/iSentence
        print "Total Training Time: %.2gs"%(time.time()-beg)