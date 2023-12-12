python main.py --method gat --dataset citeseer
python main.py --method gat --dataset cora
python main.py --method gat --dataset pubmed
python main.py --method gat --dataset amazon-photo
python main.py --method gat --dataset coauthor-cs
python main.py --method gat --dataset coauthor-physics

python main.py --method gcn --dataset citeseer
python main.py --method gcn --dataset cora
python main.py --method gcn --dataset pubmed
python main.py --method gcn --dataset amazon-photo
python main.py --method gcn --dataset coauthor-cs
python main.py --method gcn --dataset coauthor-physics

python main.py --method snexgnn --encoder gat --dataset citeseer
python main.py --method snexgnn --encoder gat --dataset cora
python main.py --method snexgnn --encoder gat --dataset pubmed
python main.py --method snexgnn --encoder gat --dataset amazon-photo
python main.py --method snexgnn --encoder gat --dataset coauthor-cs
python main.py --method snexgnn --encoder gat --dataset coauthor-physics

python main.py --method snexgnn --encoder gcn --dataset citeseer
python main.py --method snexgnn --encoder gcn --dataset cora
python main.py --method snexgnn --encoder gcn --dataset pubmed
python main.py --method snexgnn --encoder gcn --dataset amazon-photo
python main.py --method snexgnn --encoder gcn --dataset coauthor-cs
python main.py --method snexgnn --encoder gcn --dataset coauthor-physics


python main.py --method simplehgn --dataset imdb
python main.py --method simplehgn --dataset dblp
python main.py --method simplehgn --dataset acm


python main.py --method snexhgn --encoder simplehgn --dataset imdb
python main.py --method snexhgn --encoder simplehgn --dataset dblp
python main.py --method snexhgn --encoder simplehgn --dataset acm


python main.py --method snexgnn --encoder gat --dataset citeseer --eval_explanation True
python main.py --method snexgnn --encoder gat --dataset cora --eval_explanation True
python main.py --method snexgnn --encoder gat --dataset pubmed --eval_explanation True
python main.py --method snexgnn --encoder gat --dataset amazon-photo --eval_explanation True
python main.py --method snexgnn --encoder gat --dataset coauthor-cs --eval_explanation True
python main.py --method snexgnn --encoder gat --dataset coauthor-physics --eval_explanation True

python main.py --method snexgnn --encoder gcn --dataset citeseer --eval_explanation True
python main.py --method snexgnn --encoder gcn --dataset cora --eval_explanation True
python main.py --method snexgnn --encoder gcn --dataset pubmed --eval_explanation True
python main.py --method snexgnn --encoder gcn --dataset amazon-photo --eval_explanation True
python main.py --method snexgnn --encoder gcn --dataset coauthor-cs --eval_explanation True
python main.py --method snexgnn --encoder gcn --dataset coauthor-physics --eval_explanation True

