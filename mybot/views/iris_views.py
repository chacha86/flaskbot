from flask import Blueprint, request, jsonify
import json

bp = Blueprint('iris', __name__, url_prefix='/iris')


@bp.route('/get-data')
def test() :

    hong = {}
    hong["이름"] = "이순신"
    hong["거주지"] = "서울"
    hong["나이"] = 33

    json_str = json.dumps(hong, ensure_ascii=False)

    return json_str







@bp.route('/predict')
def predict() :

    # s_len = int(request.args.get("sepal_length"))
    # s_wid = int(request.args.get("sepal_width"))
    # p_len = int(request.args.get("petal_length"))
    # p_wid= int(request.args.get("petal_width"))

    from sklearn.datasets import load_iris

    iris = load_iris()  # 붓꽃 데이터

    data = iris.data  # 학습용 데이터
    target = iris.target
    feature_names = iris.feature_names  # 특성명
    target_names = iris.target_names  # 타겟명

    from sklearn.model_selection import train_test_split

    trd, tsd, trt, tst = train_test_split(data, target, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(max_depth=5, n_estimators=30)
    rfc.fit(trd, trt)

    # pred_idx = rfc.predict([[s_len, s_wid, p_len, p_wid]])

    return '{"data": "aaa", "status": "200"}'
