from classification_model import Classifier
import torch



artifacts_path="/home/aneesh/project_codenet_repo/code_corruption/codesearch/models/last_token_segmented/java"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Classifier(artifacts_path, device)



seqs = ["while ( n < 10 )",
"import java . import",
"import java . util .* ;",
"import java . util .* ; class",
"static void printTwoOdd static",
"static void void ( int arr, x ("]


preds, probs = model.classify(seqs)

print("Preds: ", preds)
print("Probs: ", probs)