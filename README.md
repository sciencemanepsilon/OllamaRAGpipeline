# OllamaRAGpipeline
## k8s port forward command
kubectl port-forward pod/your_podname 8080:80 -n your_namespace
ctrl-c to exit
## curl command from second terminal
curl -X POST http://127.0.0.1:8080/path?param=value
-H "Content-Type: application/json"
-d '{"k1": "Ada", "k2": [55,70], "k3":"john"}'