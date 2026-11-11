from flask import Flask, request, jsonify
from methods import SplitterBase, TableSplitters, TextSplitters
app = Flask(__name__)


class EndpointBase:
    method = {
        'nlp': TextSplitters.NlpMethod,
        'TextLangChain': TextSplitters.LangChainMethod,
        'TableSimple': TableSplitters.Simple,
        'TableLangChain': TableSplitters.LangChainMethod
    }
    @classmethod
    def Run(cls, item, data, i, j, label):
        method = cls.method[label]
        print(f"START {method.__name__} {i}/{j} {item} {data}")
        StrList, RecList = method(data['text'], i, j, item, data['headings'])
        print(f"StringList: {len(StrList)} RecordList: {len(RecList)}\nEND")
        return StrList, RecList



@app.route('/ChunkRecord/<i>/<j>', methods=['POST'])
def ChunkRecord(i, j):
    item = request.args.get('item')
    data = request.get_json()
    print(f"START ChunkRecord {i}/{j} {item} {data}\nEND")
    return jsonify(SplitterBase.ChunkRecord(
        i, j, data['text'], item, data['headings'], data['SplitMeth']
    )), 200


@app.route('/Splitter/<label>/<i>/<j>', methods=['POST'])
def TableSimple(label, i, j):
    StringList, RecordList = EndpointBase.Run(
        request.args.get('item'), request.get_json(), i, j, label
    )
    return jsonify(StringList, RecordList), 200