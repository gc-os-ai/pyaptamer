from pyaptamer.benchmarking.preprocessors import BasePreprocessor


class AptaTransPreprocessor(BasePreprocessor):
    def _transform(self, df):
        df = df.copy()
