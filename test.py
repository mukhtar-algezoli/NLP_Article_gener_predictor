from NewsDataset import *
from NewsClassifier import *
from Dataloader import *
from loadGlove import *
from argparse import Namespace
import torch
import torch.optim as optim
from tqdm import tqdm_notebook
import os

PATH = "savedmodel.tar"

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

args = Namespace(
    # Data and Path hyper parameters
    news_csv="ag_news/news_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch5/document_classification",
    # Model hyper parameters
    glove_filepath='glove_data/glove.6B.100d.txt', 
    use_glove=True,
    embedding_size=100, 
    hidden_dim=100, 
    num_channels=100, 
    # Training hyper parameter
    seed=1337, 
    learning_rate=0.001, 
    dropout_p=0.1, 
    batch_size=128, 
    num_epochs=100, 
    early_stopping_criteria=5, 
    # Runtime option
    cuda=False, 
    catch_keyboard_interrupt=True, 
    reload_from_files=False,
    expand_filepaths_to_save_dir=True
)


def make_train_state(args):
    return {'epoch_index':0,
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[],
            'test_loss':-1,
            'test_acc':-1}
train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# create dataset and vectorizer
dataset = NewsDataset.load_dataset_and_make_vectorizer(args.news_csv)
vectorizer = dataset.get_vectorizer()

if args.use_glove:
    words = vectorizer.title_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath, 
                                       words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None

classifier = NewsClassifier(embedding_size=args.embedding_size, 
                            num_embeddings=len(vectorizer.title_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim, 
                            num_classes=len(vectorizer.category_vocab), 
                            dropout_p=args.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0)
vectorizer = dataset.get_vectorizer()


classifier = classifier.to(args.device)

dataset.class_weights = dataset.class_weights.to(args.device)
    
loss_fuc = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)


                                           
train_state = make_train_state(args)

if os.path.exists(PATH):
  checkpoint = torch.load(PATH)
  classifier.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print("checkpoint loaded")
# epoch_bar = tqdm_notebook(desc='training routine', 
                          # total=args.num_epochs,
                          # position=0)

# dataset.set_split('train')
# train_bar = tqdm_notebook(desc='split=train',
                          # total=dataset.get_num_batches(args.batch_size), 
                          # position=1, 
                          # leave=True)
# dataset.set_split('val')
# val_bar = tqdm_notebook(desc='split=val',
                        # total=dataset.get_num_batches(args.batch_size), 
                        # position=1, 
                        # leave=True)
                        
if True:
    dataset.set_split('test')
    batch_generator = generate_batches(dataset , batch_size = args.batch_size , device = args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()
    for batch_index,batch_dict in enumerate(batch_generator):
        # print("batch num: " + str(batch_index))
        print(batch_index)
        y_pred = classifier(x_in = batch_dict['x_data'])
        loss = loss_fuc(y_pred , batch_dict['y_target'])
        loss_batch = loss.item()
        running_loss += (loss_batch-running_loss)/(batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
         # train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  # # epoch=epoch_index)
        # train_bar.update()        
    print( "eval_running_loss:" + str(round(running_loss , 4)) + "    " + "eval_running_acc:%" + str(round(running_acc , 2)))



