from optparse import OptionParser, OptionGroup
from arc_hybrid import ArcHybridLSTM
from options_manager import OptionsManager
import pickle, utils, os, time, sys, copy, itertools, re, random
from shutil import copyfile
import codecs
import csv
import operator

top_k_dict = {} # keep populating this dictionary if we have a higher dev score, if the number of entries exceeds 'k', overwrite the minimum k,v pair.

def predict_and_eval_dev(parser, om, options, cur_treebank, epoch):

    if options.multiling:
        pred_langs = [lang for lang in om.languages if lang.pred_dev] # languages which have dev data on which to predict
        for lang in pred_langs:
            lang.outfilename = os.path.join(lang.outdir, 'dev_epoch_' + str(epoch) + '.conllu')
            print "Predicting on dev data for " + lang.name
        devdata = list(utils.read_conll_dir(pred_langs, "dev"))
        if options.fingerprint and epoch == options.first_epoch:
            parser.print_data_fingerprint(devdata, 'dev data')
            # re-set devdata as it can only be iterated once
            devdata = list(utils.read_conll_dir(pred_langs, "dev"))
        pred = list(parser.Predict(devdata, om, options))
        if len(pred) > 0:
            utils.write_conll_multiling(pred, pred_langs)
        else:
            print "Warning: prediction empty"
        if options.pred_eval:
            for lang in pred_langs:
                print "Evaluating dev prediction for " + lang.name
                utils.evaluate(lang.dev_gold, lang.outfilename, om.conllu)
    else: # monolingual case
        if cur_treebank.pred_dev:
            print "Predicting on dev data for " + cur_treebank.name
            devdata = list(utils.read_conll(cur_treebank.devfile, cur_treebank.iso_id))
            if options.fingerprint and epoch == options.first_epoch:
                parser.print_data_fingerprint(devdata, 'dev data')
            cur_treebank.outfilename = os.path.join(cur_treebank.outdir, 'dev_epoch_' + str(epoch) + ('.conll' if not om.conllu else '.conllu')) # outdir was a global variable
            pred = list(parser.Predict(devdata, om, options))
            utils.write_conll(cur_treebank.outfilename, pred)
            if options.pred_eval:
                print "Evaluating dev prediction for " + cur_treebank.name
                score = utils.evaluate(cur_treebank.dev_gold, cur_treebank.outfilename, om.conllu)
                if options.model_selection:
                    if score > cur_treebank.dev_best[1]:
                        cur_treebank.dev_best = [epoch, score]
                        if options.top_k_epochs: #TODO try Python's heapq module
                            top_k_dict[epoch] = score
                            model_file = os.path.join(cur_treebank.outdir, options.model + str(epoch)) # write in the first few models, they will be overwritten later
                            parser.Save(model_file)
                            if len(top_k_dict) == options.top_k_epochs + 1: # + 1 because we're going to delete an entry so we are left with k after adding the new element.
                                print 'Reached k model entries, clearing lowest entry'
                                sorted_top_k = sorted(top_k_dict.items(), key=operator.itemgetter(1)) # sort by second element in tuple (LAS).
                                min_entry = sorted_top_k[0]
                                min_epoch, min_score = min_entry[0], min_entry[1]
                                del top_k_dict[min_epoch]
                                min_model_file = os.path.join(cur_treebank.outdir, options.model + str(min_epoch)) # remove the model we have just cleared from the dict.
                                if os.path.exists(min_model_file):
                                    os.remove(min_model_file)
                    else: # we need to overwrite lower entries in the dictionary, e.g. in the case that dev is not higher than current best dev but is higher than lowest dict entry.
                        if options.top_k_epochs:
                            sorted_top_k = sorted(top_k_dict.items(), key=operator.itemgetter(1))
                            min_entry = sorted_top_k[0]
                            min_epoch, min_score = min_entry[0], min_entry[1]
                            if score > min_score: # we can assume we already have a value for min_score as the if statement above will always be true for the first epoch.
                                top_k_dict[epoch] = score # add new element to dictionary
                                del top_k_dict[min_epoch]
                                model_file = os.path.join(cur_treebank.outdir, options.model + str(epoch))
                                parser.Save(model_file)
                                min_model_file = os.path.join(cur_treebank.outdir, options.model + str(min_epoch))
                                if os.path.exists(min_model_file):
                                    os.remove(min_model_file)
    sys.stdout.flush()


def run(om,options,i):

    if options.multiling:
        outdir = options.outdir # cur_treebank = None ?
        cur_treebank = None
    else:
        cur_treebank = om.languages[i]
        outdir = cur_treebank.outdir

    if options.shared_task:
        outdir = options.shared_task_outdir

    if not options.predict: # training
        print 'Preparing vocab'
        if options.multiling:
            words, w2i, pos, cpos, rels, langs, ch = utils.vocab(om.languages, path_is_dir=True)
        else:
            words, w2i, pos, cpos, rels, langs, ch = utils.vocab(cur_treebank.trainfile)

        paramsfile = os.path.join(outdir, options.params)
        with open(paramsfile, 'w') as paramsfp:
            print 'Saving params to ' + paramsfile
            pickle.dump((words, w2i, pos, rels, cpos, langs,
                         options, ch), paramsfp)
            print 'Finished collecting vocab'

        print 'Initializing blstm arc hybrid:'
        parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                               ch, options)

        durations = []
        for epoch in xrange(options.first_epoch, options.first_epoch+options.epochs):

            print 'Starting epoch ' + str(epoch)
            start_time = time.time()

            if options.multiling:
                traindata = list(utils.read_conll_dir(om.languages, "train", options.max_sentences))
            else:
                traindata = list(utils.read_conll(cur_treebank.trainfile, cur_treebank.iso_id,options.max_sentences))


            parser.Train(traindata, om, options)
            print 'Finished epoch ' + str(epoch)

            if not options.top_k_epochs: # save a model for each epoch as normal
                model_file = os.path.join(outdir, options.model + str(epoch))
                parser.Save(model_file)

            if options.pred_dev: # use the model to predict on dev data
                predict_and_eval_dev(parser, om, options, cur_treebank, epoch)

            if options.deadline:
                # keep track of duration of training+eval
                now = time.time()
                duration = now - start_time
                durations.append(duration)
                # estimate when next epoch will finish
                last_five_durations = durations[-5:]
                eta = time.time() + max(last_five_durations)
                print 'Deadline in %.1f seconds' %(options.deadline-now)
                print 'ETA of next epoch in %.1f seconds' %(eta-now)
                # does it exceed the deadline?
                exceeds_deadline = eta > options.deadline
            else:
                # no deadline
                exceeds_deadline = False

            if exceeds_deadline or epoch == options.epochs:
                # at the last epoch copy the best model to barchybrid.model
                if not options.model_selection:
                    # model selection off completely (for example multilingual case)
                    # --> take the final epoch, i.e. the current epoch
                    best_epoch = epoch
                else:
                    best_epoch = cur_treebank.dev_best[0] # will be final epoch by default if model selection not on for this treebank
                    if cur_treebank.model_selection:
                        print "Best dev score of " + str(cur_treebank.dev_best[1]) + " found at epoch " + str(cur_treebank.dev_best[0])

                bestmodel_file = os.path.join(outdir,"barchybrid.model" + str(best_epoch))
                model_file = os.path.join(outdir,"barchybrid.model")
                print "Copying " + bestmodel_file + " to " + model_file
                copyfile(bestmodel_file,model_file)

            if exceeds_deadline and epoch < options.epochs:
                print 'Leaving epoch loop early to avoid exceeding deadline'
                break

            if exceeds_deadline and epoch < options.epochs:
                print 'Leaving epoch loop early to avoid exceeding deadline'
                break

    else: #if predict - so

        if options.multiling:
            modeldir = options.modeldir
        else:
            modeldir = om.languages[i].modeldir

        params = os.path.join(modeldir,options.params)
        print 'Reading params from ' + params
        with open(params, 'r') as paramsfp:
            words, w2i, pos, rels, cpos, langs, stored_opt, ch = pickle.load(paramsfp)

            parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                               ch, stored_opt)
            model = os.path.join(modeldir, options.model)
            parser.Load(model)

            if options.multiling:
                testdata = utils.read_conll_dir(om.languages,"test")
            else:
                testdata = utils.read_conll(cur_treebank.testfile,cur_treebank.iso_id)

            ts = time.time()

            if options.pred_eval:
                if options.multiling:
                    if om.weighted_tb and om.tb_weights: # write values to csv for analysis.
                        tbidmetadata = parser.tbidmetadata # load weights and tbid values
                        print tbidmetadata.items()
                        filename = os.path.join(outdir, 'multi_LAS_scores.csv')
                        needs_header = not os.path.exists(filename)
                        with open(filename, 'a') as f:
                            writer = csv.writer(f)
                            if needs_header:
                                print "Creating a new weighted tbemb csv file in outdir and writing in headers."
                                tbids = []
                                weights = []
                                scores = []
                                for tbm_index in range(len(tbidmetadata.keys())):
                                    tbids.append('tbid%d' %(tbm_index+1))
                                    weights.append('weight%d' %(tbm_index+1))
                                    scores.append('las-on-tb%d' %(tbm_index+1))
                                writer.writerow(tbids + weights + scores)
                            for tbid, v in tbidmetadata.items():
                                for l in om.languages:
                                    if l.name in v:
                                        print "Evaluating on " + l.name
                                        score = utils.evaluate(l.test_gold, l.outfilename, om.conllu)
                                        print "Obtained LAS F1 score of %.2f on %s" % (score, l.name)
                                        tbidmetadata[tbid].append('%.9f' %score) # append new LAS score to our dictionary
                            print tbidmetadata.items()
                            tbids = []
                            weights = []
                            scores = []
                            for tbid, v in sorted(tbidmetadata.items()):
                                if len(v) < 4:
                                    v.append('-')
                                tbname, weight, tbindex, score = v
                                tbids.append(tbid)
                                weights.append(weight)
                                scores.append(score)
                            weight_data = tbids + weights + scores
                            print weight_data
                            writer.writerow(weight_data)
                    else:
                        for l in om.languages:
                            print "Evaluating on " + l.name
                            score = utils.evaluate(l.test_gold,l.outfilename,om.conllu)
                            print "Obtained LAS F1 score of %.2f on %s" %(score,l.name)
                else:
                    print "Evaluating on " + cur_treebank.name
                    score = utils.evaluate(cur_treebank.test_gold,cur_treebank.outfilename,om.conllu)
                    print "Obtained LAS F1 score of %.2f on %s" %(score,cur_treebank.name)

            print 'Finished predicting'

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--outdir", metavar="PATH", help='Output directory')
    parser.add_option("--datadir", metavar="PATH",
        help="Input directory with UD train/dev/test files; obligatory if using --include")
    parser.add_option("--modeldir", metavar="PATH",
        help='Directory where models will be saved, defaults to same as --outdir if not specified')
    parser.add_option("--params", metavar="FILE", default="params.pickle", help="Parameters file")
    parser.add_option("--model", metavar="FILE", default="barchybrid.model",
        help="Load/Save model file")

    group = OptionGroup(parser, "Experiment options")
    group.add_option("--include", metavar="LIST", help="List of languages by ISO code to be run \
if using UD. If not specified need to specify trainfile at least. When used in combination with \
--multiling, trains a common parser for all languages. Otherwise, train monolingual parsers for \
each")
    group.add_option("--treebanks-from-json", action="store_true", default=False,
        help='Read available treebanks from src/utils/ud_iso.json (or metadata.json in shared \
task mode) rather than scanning datadir.')
    group.add_option("--trainfile", metavar="FILE", help="Annotated CONLL(U) train file")
    group.add_option("--devfile", metavar="FILE", help="Annotated CONLL(U) dev file")
    group.add_option("--testfile", metavar="FILE", help="Annotated CONLL(U) test file")
    group.add_option("--epochs", type="int", metavar="INTEGER", default=30,
        help='Number of epochs')
    group.add_option("--deadline", type="float", metavar="FLOAT", default=0.0,
        help='Do not start another epoch if ETA is after the provided deadline \
(seconds after the system epoch, usually 1st of Jan 1970; default = no deadline)')
    group.add_option("--predict", help='Parse', action="store_true", default=False)
    group.add_option("--multiling", action="store_true", default=False,
        help='Train a multilingual parser with language embeddings')
    group.add_option("--max-sentences", type="int", metavar="INTEGER",
        help='Only train using n sentences per epoch', default=-1)
    group.add_option("--create-dev", action="store_true", default=False,
        help='Create dev data if no dev file is provided')
    group.add_option("--min-train-sents", type="int", metavar="INTEGER", default=1000,
        help='Minimum number of training sentences required in order to create a dev file')
    group.add_option("--dev-percent", type="float", metavar="FLOAT", default=5,
        help='Percentage of training data to use as dev data')
    group.add_option("--top-k-epochs", type="int", metavar="INTEGER", default=0,
                     help='Only save k number of models based on dev LAS score (to save storage space)')
    group.add_option("--disable-pred-dev", action="store_false", dest="pred_dev", default=True,
        help='Disable prediction on dev data after each epoch')
    group.add_option("--disable-pred-eval", action="store_false", dest="pred_eval", default=True,
        help='Disable evaluation of prediction on dev data')
    group.add_option("--disable-model-selection", action="store_false",
        help="Disable choosing of model from best/last epoch", dest="model_selection", default=True)
    group.add_option("--overwrite-model", action="store_true",
        help="Overwrite model if the dev score is higher than previous best dev score", dest="overwrite_model", default=False)
    group.add_option("--use-default-seed", action="store_true",
        help="Use default random seed for Python", default=False)
    #TODO: reenable this
    group.add_option("--continue", dest="continueTraining", action="store_true", default=False)
    group.add_option("--continueModel", metavar="FILE",
        help="Load model file, when continuing to train a previously trained model")
    group.add_option("--first-epoch", type="int", metavar="INTEGER", default=1)
    group.add_option("--fingerprint", action="store_true", help="Log fingerprints of all data (slow)", default=False)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Parser options")
    group.add_option("--disable-oracle", action="store_false", dest="oracle", default=True,
        help='Use the static oracle instead of the dynamic oracle')
    group.add_option("--disable-head", action="store_false", dest="headFlag", default=True,
        help='Disable using the head of word vectors fed to the MLP')
    group.add_option("--disable-rlmost", action="store_false", dest="rlMostFlag", default=True,
        help='Disable using leftmost and rightmost dependents of words fed to the MLP')
    group.add_option("--userl", action="store_true", dest="rlFlag", default=False)
    group.add_option("--k", type="int", metavar="INTEGER", default=3,
        help="Number of stack elements to feed to MLP")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Neural network options")
    group.add_option("--dynet-seed", type="int", metavar="INTEGER",
        help="Random seed for Dynet")
    group.add_option("--dynet-mem", type="int", metavar="INTEGER",
        help="Memory to assign Dynet in MB", default=512)
    group.add_option("--learning-rate", type="float", metavar="FLOAT",
        help="Learning rate for neural network optimizer", default=0.001)
    group.add_option("--char-emb-size", type="int", metavar="INTEGER",
        help="Character embedding dimensions", default=24)
    group.add_option("--char-lstm-output-size", type="int", metavar="INTEGER",
        help="Character BiLSTM dimensions", default=50)
    group.add_option("--word-emb-size", type="int", metavar="INTEGER",
        help="Word embedding dimensions", default=100)
    group.add_option("--lang-emb-size", type="int", metavar="INTEGER",
        help="Language embedding dimensions", default=12)
    group.add_option("--elmo-emb-size", type="int", metavar="INTEGER", 
        help="ELMo embedding dimensions", default=1024, dest='elmo_emb_size')
    group.add_option("--lstm-output-size", type="int", metavar="INTEGER",
        help="Word BiLSTM dimensions", default=125)
    group.add_option("--mlp-hidden-dims", type="int", metavar="INTEGER",
        help="MLP hidden layer dimensions", default=100)
    group.add_option("--mlp-hidden2-dims", type="int", metavar="INTEGER",
        help="MLP second hidden layer dimensions", default=0)
    group.add_option("--external-embedding", metavar="FILE", help="External embeddings")
    group.add_option("--activation", help="Activation function in the MLP", default="tanh")
    group.add_option("--disable-bilstm", action="store_true", default=False,
        help='disable the BiLSTM feature extactor')
    group.add_option("--disable-lembed", action="store_false", dest="use_lembed",
        help='disable the use of a language embedding when in multilingual mode', default=True)
    group.add_option('--use-elmo', action='store_true', dest='use_elmo', default=False,
        help='use ELMo representations for the data')
    group.add_option('--elmo-script-dir', type="str", default=None, dest='elmo_script_dir',
        help='location of gen_elmo.sh')
    group.add_option('--elmo-layer', type="str", default='average', dest='elmo_layer',
        help='which elmo layer to use: average, all')
    group.add_option('--elmo-output-dir', metavar='PATH', default=None, dest='elmo_output_dir',
        help='location of the pre-computed ELMo representations')
    group.add_option("--weighted-tbemb", action="store_true", dest="weighted_tb", default=False,
        help='flag to enable weighted tbemb')
    group.add_option("--tb-weights", type="string", dest="tb_weights", default='',
        help='specify tbid weights in the format "tbidx:weight" ')
    group.add_option("--tb-weights-from-file", action="store_true", dest="tb_weights_from_file", default=False,
        help="read tbemb sentence-level weights from .tbweights file")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Debug options")
    group.add_option("--debug", action="store_true",
        help="Run parser in debug mode, with fewer sentences", default=False)
    group.add_option("--debug-train-sents", type="int", metavar="INTEGER",
        help="Number of training sentences in --debug mode", default=150)
    group.add_option("--debug-dev-sents", type="int", metavar="INTEGER",
        help="Number of dev sentences in --debug mode", default=100)
    group.add_option("--debug-test-sents", type="int", metavar="INTEGER",
        help="Number of test sentences in --debug mode", default=50)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Shared task options")
    group.add_option("--shared_task", action="store_true", dest="shared_task", default=False)
    group.add_option("--shared_task_datadir", dest="shared_task_datadir", default="EXP")
    group.add_option("--shared_task_outdir", dest="shared_task_outdir", default="EXP")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    # really important to do this before anything else to make experiments reproducible
    utils.set_seeds(options)

    om = OptionsManager(options)
    for i in range(om.iterations):
        run(om,options,i)