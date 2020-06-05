from code.lib import *
from code.config import config
from code.logger import log


def seed_everything(seed):
    log.info(f'seed: {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_csv(file_path):
    data = pd.read_csv(file_path)
    bboxs = np.stack(data['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        data[column] = bboxs[:,i]
    data.drop(columns=['bbox'], inplace=True)
    return data


def kfold(data):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = data[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = data[['image_id', 'source']].groupby('image_id').min()['source']

    # Why choose 15?
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )

    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    return df_folds