from ..smp import *
from ..utils import *
from .image_base import ImageBaseDataset

        
class AnomalyYorn(ImageBaseDataset):
    TYPE = "Y/N"
    DATASET_URL = {"AnomalyYorn":"/home/bhu/LMUData/AnomalyYorn.tsv"}
    DATASET_MD5 = {}
    def build_prompt(self,line):
    # tsv应当保存一个few-shot的路径 可以由这个路径随机抽取normal的图像
        if isinstance(line, int):
            line = self.data.iloc[line]
        # 考虑few-shot的情况
        # extensions_list = ['png','jpg','jpeg']
        # msgs = []
        # if not self.few_shot:
        #     few_shot_path = line['few_shot_path']
        #     normal_imgs = [os.path.join(few_shot_path,img) for img in os.listdir(few_shot_path) if img.split('.')[-1] in extensions_list] 
        #     normal_imgs = random.sample(normal_imgs,k=min(self.few_shot,len(normal_imgs))) 
        #     msgs.extend([dict(type = "image" , value = img) for img in normal_imgs])
        
        if self.meta_only:
            # 如果这里是一个包含多个路径的列表，就是一种few-shot?
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
    
    def evaluate(self,eval_file,**judge_kwargs):
        from datetime import datetime
        from .utils.yorn import YOrN_Extraction, YOrN_auxeval, Anomaly_classification_rating
        from .utils.yorn import default_rating, MME_rating, Hallusion_rating, POPE_rating, AMBER_rating
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        suffix = eval_file.split('.')[-1]
        time_stamp = datetime.now().timestamp()
        storage = eval_file.replace(f'.{suffix}', f'{time_stamp}.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        
        if not osp.exists(storage):
            ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == 'Unknown' and tmp[k] != 'Unknown':
                        ans_map[k] = tmp[k]

            data['extracted'] = [ans_map[x] for x in data['index']]
            unknown = data[data['extracted'] == 'Unknown']
            # 没能提供判断答案所需的api 就是精确判断模式
            model = judge_kwargs.get('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')

            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown['index'])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data['extracted'] = [ans_map[x] for x in data['index']]
            dump(data, storage)

        data = load(storage)
        data['score'] = (data['answer'].str.lower() == data['extracted'].str.lower())
        dump(data, storage)
        score = Anomaly_classification_rating(storage)
        score_tgt = eval_file.replace('.xlsx', '_score.csv')
        dump(score, score_tgt)
        return score