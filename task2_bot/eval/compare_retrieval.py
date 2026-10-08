"""Офлайн dev-сравнение поиска. Только локальные эмбеддинги, без BardBorn/OpenAI-compatible API."""
import argparse
import csv
import json
from pathlib import Path
from task2_bot.config import ROOT, INDEX
from task2_bot.rag.embeddings import Embedder
from task2_bot.rag.search import Retriever
from task2_bot.eval.retrieval import evaluate, load_questions


class CachedEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}

    def query(self, text):
        if text not in self.cache:
            self.cache[text] = self.embedder.query(text)
        return self.cache[text]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=INDEX)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'dev_ablation')
    args=parser.parse_args()
    manifest=json.loads((args.index/'manifest.json').read_text())
    encoder=CachedEmbedder(Embedder(model=manifest['model']))
    questions=load_questions(ROOT/'eval'/'questions.json','dev')
    # Equal question set, same chunks; vary one axis at a time against baselines.
    configs=[(m,t,s,.25) for m in ('bm25','semantic','hybrid') for t in (False,True) for s in (True,False)]
    configs += [('hybrid',t,True,1.) for t in (False,True)]
    args.output.mkdir(parents=True,exist_ok=True)
    table=[]
    for mode,translated,scope,weight in configs:
        name=f'{mode}_translation-{int(translated)}_scope-{int(scope)}_weight-{weight}'
        retriever=Retriever(args.index,embedder=encoder,mode=mode,use_scope=scope,lexical_weight=weight)
        result=evaluate(retriever,questions,translated=translated)
        result['configuration']=dict(split='dev',mode=mode,translation='manual' if translated else 'off',scope=scope,
            lexical_weight=weight,model=manifest['model'],chunks_sha256=manifest['chunks_sha256'])
        (args.output/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        row=dict(configuration=name,**result['summary'])
        table.append(row)
        print(f'{name}: {row["page_hits"]}/{row["n"]}; MRR={row["mrr"]:.3f}',flush=True)
    with (args.output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)


if __name__=='__main__': main()
