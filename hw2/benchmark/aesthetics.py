from audiobox_aesthetics.infer import initialize_predictor

class Aesthetics:
    def __init__(self):
        self.predictor = initialize_predictor()

    def get_score(self, audio_path):
        score = self.predictor.forward(
            [{"path": audio_path}]
        )[0]
        print(score)
        return (score["CE"], score["CU"], score["PC"], score["PQ"])