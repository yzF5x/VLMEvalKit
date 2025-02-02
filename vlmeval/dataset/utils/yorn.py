from ...smp import *


def AMBER_rating(data_file):
    data = load(data_file)
    stats = defaultdict(dict)
    lt = len(data)
    category_mapping = {
        'discriminative-attribute-state': 'Attribute',
        'discriminative-attribute-number': 'Attribute',
        'discriminative-attribute-action': 'Attribute',
        'discriminative-hallucination': 'Existence',
        'discriminative-relation': 'Relation',
        'relation': 'Relation'
    }

    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']

        new_category = category_mapping.get(category, category)

        if image_path not in stats[new_category]:
            stats[new_category][image_path] = []
        stats[new_category][image_path].append(score)

    def acc(key):
        res = stats[key]
        values = []
        for val in res.values():
            values.extend(val)
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k)

    scores['Avg ACC'] = np.mean(list(scores.values()))
    ret = d2df(scores)
    return ret


def MME_rating(data_file):
    data = load(data_file)
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)

    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )

    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base
    ret.update(scores)
    ret = d2df(ret)
    return ret


def Hallusion_rating(data_file):
    def calc_fAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_qAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_aAcc(data):
        return np.mean(data['score']) * 100

    data = load(data_file)
    data['set_id'] = [x.split('_')[3] for x in data['index']]
    data['figure_id'] = [x.split('_')[4] for x in data['index']]
    data['question_id'] = [x.split('_')[5] for x in data['index']]

    res = dict(split=[], aAcc=[], fAcc=[], qAcc=[])
    res['split'].append('Overall')
    res['aAcc'].append(calc_aAcc(data))
    res['fAcc'].append(calc_fAcc(data))
    res['qAcc'].append(calc_qAcc(data))

    if 'category' in data:
        cates = list(set(data['category']))
        for c in cates:
            sub = data[data['category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))

    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        for c in cates:
            sub = data[data['l2-category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))
    ret = pd.DataFrame(res)
    return ret

def Anomaly_classification_rating(data_file):
    def cal_f1_score(y_true, y_pred):
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score, precision, recall
    data = load(data_file)
    data = data.assign(category=data['category'].str.split(',')).explode('category')
    data['index'] = range(len(data))
    res = dict(split=[], Overall=[], acc=[], precision=[], recall=[],anomalyAcc = [])
    y_true , y_pred = [], []
    anomaly_true ,anomaly_pred = [] , []
    for i in data['answer']:
        i = i.strip().replace("'",'"')
        i = json.loads(i)
        print("answer i :" ,i)
        anomaly_true.append(i.get('Anomaly'))
        i = i['standards']
        for key , answer in i.items():
            if answer.lower() == 'yes':
                y_true.append(1)
            else:
                y_true.append(0)
    for i in data['extracted']:
        print("extracted i :" ,i)
        i = json.loads(i)
        # print("extracted i :" ,i)
        anomaly_pred.append(i.get('anomaly') if not i.get('anomaly') else 'W')
        i = i['standards']
        for key , pred in i.items():
            if pred.get('answer') =='Yes':
                y_pred.append(1)
            else:
                y_pred.append(0)
    correct_count = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
    total_count = len(y_pred)
    res['acc'].append(correct_count/total_count * 100)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_score, precision, recall = cal_f1_score(y_true, y_pred)
    res['split'].append('Overall')
    res['Overall'].append(f1_score * 100)
    res['precision'].append(precision * 100)
    res['recall'].append(recall * 100)
    
    if len(anomaly_pred) == len(anomaly_true) and len(anomaly_pred):
        correct = [1 if anomaly_pred[i] == anomaly_true[i] else 0 for i in range(len(anomaly_pred))]
        correct_count = sum(correct)
        total_count = len(anomaly_true)
        res['anomalyAcc'].append(correct_count/total_count * 100)
    else:
        print("true: ",anomaly_true ,"pred: ", anomaly_pred)
        res['anomalyAcc'].append(-1)
    print('Overall:' , res)
    
    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        for c in cates:
            sub = data[data['category'] == c]
            y_true , y_pred = [], []
            anomaly_true ,anomaly_pred = [] , []
            for i in sub['answer']:
                i = i.strip().replace("'",'"')
                i = json.loads(i)
                print("answer i :" ,i)
                anomaly_true.append(i.get('Anomaly'))
                i = i['standards']
                for key , answer in i.items():
                    if answer.lower() == 'yes':
                        y_true.append(1)
                    else:
                        y_true.append(0)
            for i in sub['extracted']:
                print("extracted i :" ,i)
                i = json.loads(i)
                # print("extracted i :" ,i)
                anomaly_pred.append(i.get('anomaly') if not i.get('anomaly') else 'W')
                i = i['standards']
                for key , pred in i.items():
                    if pred.get('answer') =='Yes':
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            correct_count = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
            total_count = len(y_pred)
            res['acc'].append(correct_count/total_count * 100)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            f1_score, precision, recall = cal_f1_score(y_true, y_pred)
            res['split'].append(c)
            res['Overall'].append(f1_score * 100)
            res['precision'].append(precision * 100)
            res['recall'].append(recall * 100)
            if len(anomaly_pred) == len(anomaly_true) and len(anomaly_pred):
                correct = [1 if anomaly_pred[i] == anomaly_true[i] else 0 for i in range(len(anomaly_pred))]
                correct_count = sum(correct)
                total_count = len(anomaly_true)
                res['anomalyAcc'].append(correct_count/total_count * 100)
            else:
                print("true: ",anomaly_true ,"pred: ", anomaly_pred)
                res['anomalyAcc'].append(-1)

    ret = pd.DataFrame(res)
    return ret

def POPE_rating(data_file):
    def cal_f1_score(y_true, y_pred):
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score, precision, recall

    data = load(data_file)
    data = data.assign(category=data['category'].str.split(',')).explode('category')
    data['index'] = range(len(data))
    res = dict(split=[], Overall=[], acc=[], precision=[], recall=[])
    y_true = np.array([1 if i == 'Yes' else 0 for i in data['answer']])
    y_pred = np.array([1 if i == 'Yes' else 0 for i in data['extracted']])
    f1_score, precision, recall = cal_f1_score(y_true, y_pred)
    res['split'].append('Overall')
    res['Overall'].append(f1_score * 100)
    res['acc'].append(np.mean(data['score']) * 100)
    res['precision'].append(precision * 100)
    res['recall'].append(recall * 100)

    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        for c in cates:
            sub = data[data['category'] == c]
            y_true = np.array([1 if i == 'Yes' else 0 for i in sub['answer']])
            y_pred = np.array([1 if i == 'Yes' else 0 for i in sub['extracted']])
            f1_score, precision, recall = cal_f1_score(y_true, y_pred)
            res['split'].append(c)
            res['Overall'].append(f1_score * 100)
            res['acc'].append(np.mean(sub['score']) * 100)
            res['precision'].append(precision * 100)
            res['recall'].append(recall * 100)

    ret = pd.DataFrame(res)
    return ret


def default_rating(data_file):
    data = load(data_file)
    res = {}
    res['Overall'] = np.mean(data['score']) * 100
    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        cates.sort()
        for c in cates:
            sub = data[data['category'] == c]
            res[c] = np.mean(sub['score']) * 100
    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        cates = [c for c in cates if not pd.isna(c)]
        cates.sort()
        for c in cates:
            sub = data[data['l2-category'] == c]
            res[c] = np.mean(sub['score']) * 100
    ret = d2df(res)
    return ret


def YOrN_match_prompt(line):
    tmpl = (
        'You are an AI assistant who will help me to match an answer with two options of a question. '
        'The options are only Yes / No. '
        'You are provided with a question and an answer, '
        'and you need to find which option (Yes / No) is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Unknown. '
        'Your should output a single word among the following 3 choices: Yes, No, Unknown.\n'
        'Example 1: \n'
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is 'Hello'.\nYour output: Yes\n"
        'Example 2: \n'
        "Question: Is the word in this image 'Hello'?\n"
        "Answer: The word in this image is not 'Hello'.\nYour output: No\n"
        'Example 3: \n'
        'Question: {}?\nAnswer: {}\nYour output: '
    )
    return tmpl.format(line['question'], line['prediction'])

def YOrN_Extraction_From_Json(output):
    def parse_json(output):
        output = output.replace("'",'"')
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        i = 0
        j = len(lines) - 1
        while i < len(lines):
            if '{' in lines[i]:
                break
            i += 1
        while j >= 0:
            if '}' in lines[j]:
                break
            j -= 1
        # 没有按照指定的json格式回答
        if j==-1 and i==len(lines):
            answer_dict = {}
            standards = {}
            for line in lines:
                if 'anomaly' in line or 'Anomaly' in line:
                    answer_dict['anomaly'] = line.split(':',1)[1]
                else:
                    answer_dict['anomaly'] = 'Unknown'
                if "reason" in line and ':' in line:  #回答了reason
                    part , reason = line.split('reason',1)
                    key , vlm_answer = part.split(':',1)
                    standards[key.strip()]={"answer": vlm_answer.strip(),"reason": reason.strip()}
                elif ':' in line:  #只回答了yes no
                    key, value = line.split(':', 1)  
                    standards[key.strip()] = {"answer": vlm_answer.strip()}
            answer_dict['standards'] = standards
            return json.dumps(answer_dict,indent = 4)
        else :
            answer_str = '\n'.join(lines[i:j+1])
            json_str = json.loads(answer_str)
            json_str = json.dumps(json_str,indent = 4)
            return json_str 
      
    answer = parse_json(output)
    if isinstance(answer, dict):
        standards = answer.get('standards')
        for key , value in standards.items():
            vlm_answer = value.get('answer').lower()
            vlm_answer = process_punctuation(vlm_answer).split()
            if 'yes' in vlm_answer and 'no' not in vlm_answer:
                standards[key]['answer'] = 'Yes'
            elif 'yes' not in vlm_answer and 'no' in vlm_answer:
                standards[key]['answer'] = 'No'
            else :
                standards[key]['answer'] = 'Unknown'
        answer['standards'] = standards
    print(f"YOrN_Extraction_From_Json(output) result: {answer} \n")
    return answer
    
    

def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def YOrN_auxeval(model, line):
    prompt = YOrN_match_prompt(line)
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = YOrN_Extraction(output)
        if ans != 'Unknown':
            return ans
    return 'Unknown'
