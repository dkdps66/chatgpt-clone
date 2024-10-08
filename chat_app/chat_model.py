from langchain_aws.chat_models import ChatBedrock

class ChatModel:
    """Chat 모델 클래스는 주어진 모델 ID로 대화 모델을 초기화하고 요청에 대한 응답을 제공합니다."""
    
    def __init__(self, model_id):
        self.chat_model = ChatBedrock(model_id=model_id)
    
    def get_response(self, prompt: str):
        """모델에 대한 요청을 보내고 응답을 받는다."""
        return self.chat_model.stream(prompt)
