from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dataclasses import dataclass
from mlflow.metrics.genai.metric_definitions import relevance, faithfulness
from loguru import logger
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain.prompts import PromptTemplate
from config.llm_config import LLMConfig
from config.elasticsearch_config import ESConfig
import os


class ModelManager:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_uri = None

    def __connect_elasticSearch(self):
        """
        Connect to Elasticsearch and initialize the vector search using the embedding model.

        Returns:
            ElasticsearchStore or None: The Elasticsearch store if successful, None otherwise.
        """
        try:
            logger.debug(f"Trying to connect to Elasticsearch at: {ESConfig.HOST}")

            elastic_vector_search = ElasticsearchStore(
                es_url=ESConfig.HOST,
                index_name=LLMConfig.EMBEDDING_INDEX_NAME(),
                embedding=self.bgeEmbeddings,
            )
            return elastic_vector_search

        except ConnectionError as ce:
            logger.debug(f"ConnectionError: Could not establish connection to Elasticsearch. Reason: {ce}, closing server")
            exit()

        except Exception as e:
            logger.debug(f"An unexpected error occurred: {e}, closing server")
            exit()
            
    def log_model_to_mlflow(self, artifact_path="model"):
        with mlflow.start_run(run_name='Model') as run:
            mlflow.set_experiment("SE_LLM_Experiment")
            components = {
                "model": self.model,
                "tokenizer": self.tokenizer,
            }
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path=artifact_path,
                task="text-generation",
                model_config=None
            )
            self.model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

class EvaluationManager:
    def __init__(self, gpt2_llm, vectordb, model_end_point):
        self.gpt2_llm = gpt2_llm
        self.vectordb = vectordb
        self.model_end_point = model_end_point

    def retrieve_doc_ids(self, question: str):
        docs = self.vectordb.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(self, question_df: pd.DataFrame) -> pd.Series:
        return question_df["question"].apply(self.retrieve_doc_ids)

    def evaluate_retriever(self, eval_data):
        with mlflow.start_run(run_name="retriever part evaluation with different values") as run:
            evaluate_results = mlflow.evaluate(
                model=self.retriever_model_function,
                data=eval_data,
                targets="source",
                evaluators="default",
                extra_metrics=[
                    mlflow.metrics.precision_at_k(1),
                    mlflow.metrics.precision_at_k(2),
                    mlflow.metrics.precision_at_k(3),
                    mlflow.metrics.recall_at_k(1),
                    mlflow.metrics.recall_at_k(2),
                    mlflow.metrics.recall_at_k(3),
                    mlflow.metrics.ndcg_at_k(1),
                    mlflow.metrics.ndcg_at_k(2),
                    mlflow.metrics.ndcg_at_k(3),
                ],
            )
            return evaluate_results.tables["eval_results_table"]

    def plot_metrics(self, evaluate_results):
        for metric_name in ["precision", "recall", "ndcg"]:
            y = [evaluate_results.metrics[f"{metric_name}_at_{k}/mean"] for k in range(1, 4)]
            plt.plot([1, 2, 3], y, label=f"{metric_name}@k")
        plt.xlabel("k")
        plt.ylabel("Metric Value")
        plt.title("Metrics Comparison at Different Ks")
        plt.xticks([1, 2, 3])
        plt.legend()
        plot_path = "mlruns/artifacts/metrics_plot.png"
        plt.savefig(plot_path)
        with mlflow.start_run(run_name="metrics plot") as run:
            mlflow.log_artifact(plot_path, artifact_path="mlruns/artifacts/metrics_plot.png")

class PhiModelManager(ModelManager):
    def log_model_to_mlflow(self, artifact_path="model"):
        with mlflow.start_run(run_name='Phi-1.5 model') as run:
            mlflow.set_experiment("SE_LLM_Evaluation")
            components = {
                "tokenizer": self.tokenizer,
                "model": self.model,
            }
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path=artifact_path,
            )
            self.model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

@dataclass
class EvaluationExample:
    input: str
    output: str
    score: int
    justification: str
    grading_context: dict

class EvaluationMetrics:
    def __init__(self, model_end_point):
        self.model_end_point = model_end_point

    def calculate_faithfulness_metric(self, examples):
        # Assuming faithfulness() and relevance() are functions defined elsewhere
        faithfulness_metric = faithfulness(model=self.model_end_point, examples=examples)
        return faithfulness_metric

    def evaluate_model(self, model, eval_df):
        with mlflow.start_run():
            results = mlflow.evaluate(
                model,
                eval_df,
                model_type="question-answering",
                evaluators="default",
                predictions="result",
                extra_metrics=[self.calculate_faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
                evaluator_config={"col_mapping": {"inputs": "questions", "context": "source_documents"}},
            )
            return results.metrics, results.tables["eval_results_table"]

# Example usage:
gpt2_model_manager = ModelManager("openai-community/gpt2")
gpt2_model_manager.log_model_to_mlflow()

phi_model_manager = PhiModelManager("microsoft/phi-1_5")
phi_model_manager.log_model_to_mlflow()

pipe = pipeline(
    'text-generation',
    model=gpt2_model_manager.model,
    tokenizer=gpt2_model_manager.tokenizer,
    max_length=1024,
)
gpt2_llm = HuggingFacePipeline(pipeline=pipe)

vectordb = Chroma(persist_directory=directory, embedding_function=bgeEmbeddings)
retriever = vectordb.as_retriever(search_type="similarity")

eval_manager = EvaluationManager(gpt2_llm, vectordb, phi_model_manager.model_uri)
evaluate_results = eval_manager.evaluate_retriever(eval_data)
eval_manager.plot_metrics(evaluate_results)

faithfulness_examples = [...]  # Define your faithfulness examples
evaluation_metrics = EvaluationMetrics(phi_model_manager.model_uri)
faithfulness_metric = evaluation_metrics.calculate_faithfulness_metric(faithfulness_examples)

model_evaluator = ModelEvaluator()
metrics, eval_results_table = model_evaluator.evaluate_model(model, eval_df)
display(eval_results_table)
