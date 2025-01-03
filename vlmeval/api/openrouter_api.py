from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
from PIL import Image

class Openrouter_Wrapper(BaseAPI):
    
    is_api: bool = True
    
    def __init__(self,
                model: str = 'google/gemini-pro-vision',
                api_base: str = 'https://openrouter.ai/api/v1/chat/completions',
                img_detail: str = 'low',
                img_size: int = -1,
                key: str = None,
                retry: int = 3,
                wait: int = 3,
                system_prompt: str = None,
                verbose: bool = True,
                temperature: float = 0,
                max_tokens: int = 1024,
                **kwargs):
        self.url = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fail_msg = 'Failed to obtain answer via API. '
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('OPENROUTER_API_KEY', '') 

        self.headers = {
            "Authorization": f"Bearer " + self.key
        }
        super().__init__(retry=retry, wait=wait, verbose=verbose, system_prompt=system_prompt, **kwargs)
    
    def prepare_inputs(self,inputs):
        # input的格式类似：
        # [
        #     {
        #         'type': 'image', 
        #         'value': '/data/yuezhu/ad-exp/AD-exp/dataset/mvtec_loco_anomaly_detection_orig/breakfast_box/test/structural_anomalies/009.png'
        #     }, 
        #     {
        #         'type': 'text', 
        #         'value': 'First,tell me \'YES\' or \'NO\' and then tell me your reason.            Your answer should be json format like:            {"vlm_answer":"Yes"or"No","reason":"..."} \n            My question is : does this image meet this standard \'The breakfast box contains exactly two tangerines, one nectarine, a section of granola, and a mix of banana chips and almonds.No foreign objects are present in the box .\'?'
        #     }
        # ]
        messages = []
        info =  {
            "type": "image_url",
            "image_url": {
                "url": "",
                "detail": self.img_detail
            }
        }
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        content = []
        for item in inputs:
            if item['type'] == 'image':
                # TODO 将图片转换为base64
                encoded_image = encode_image_to_base64(item['value'],target_size=self.img_size)
                info['image_url']['url'] = f"data:image/jpeg;base64,{encoded_image}"
                content.append(info)
            elif item['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                conten.append(item)
        messages.append(
            {'role': 'user', 'content': content}
        )
        return messages
        
    def generate_inner(self, inputs, **kwargs) -> str:
        # payload = prepare_inputs(inputs)
        messages = []
        info =  {
            "type": "image_url",
            "image_url": {
                "url": "",
                "detail": self.img_detail
            }
        }
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        content = []
        for item in inputs:
            if item['type'] == 'image':
                # TODO 将图片转换为base64
                encoded_image = encode_image_file_to_base64(item['value'],target_size=self.img_size)
                info['image_url']['url'] = f"data:image/jpeg;base64,{encoded_image}"
                content.append(info)
            if item['type'] == 'text':
                item = {'type': 'text', 'text': item['value']}
                content.append(item)
        messages.append(
            {'role': 'user', 'content': content}
        )
        payload = {
            "model":self.model,
            "messages":messages
        }
        # print("\n",messages,"\n")
        response = requests.post(url = self.url , headers = self.headers ,json = payload)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)
        self.logger.info(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&vlmanswer : {answer}")
        return ret_code, answer, response
    

class OpenrouterAPI(Openrouter_Wrapper):

    def generate(self, message, dataset=None):
        return super(Openrouter_Wrapper, self).generate(message)