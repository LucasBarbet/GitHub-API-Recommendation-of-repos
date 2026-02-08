from svd_package.train import train_svd_from_txt
from svd_package.evaluation import RecoEvaluator

model, (train_df, eval_df, test_df), _, _ = train_svd_from_txt(
    split_mode="random"   # ou "per_user"
)

evaluator = RecoEvaluator(model, train_df, top_k=5)

user = "freeborough"
reco = evaluator.recommend(user)

print(f"Recommandations pour {user}:")
for repo, score in reco:
    print(repo, "->", score)

results = evaluator.evaluate(eval_df)
print("\nRésultats évaluation :")
for k, v in results.items():
    print(k, ":", v)
