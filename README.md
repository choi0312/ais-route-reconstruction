# AIS Route Reconstruction

스마트 해운물류 × AI 미션 챌린지의 **항로 복원 예측** 문제를 기반으로 구성한 AIS 시계열 항로 복원 프로젝트입니다.


## Competition Result

| 항목 | 내용 |
|---|---|
| 대회명 | 스마트 해운물류 × AI 미션 챌린지 : AIS 데이터를 활용한 선박 항로 복원 예측 |
| 주최/주관 | 해양수산부/울산항만공사&한국정보산업연합회 |
| 문제 유형 | AIS 시계열 기반 항로 복원, 정형 시계열 예측 |
| 성과 | 2등(울산항만공사사장상) 수상|
| 대회 링크 | https://dacon.io/competitions/official/236626/overview/description |

## 1. 프로젝트 개요

본 문제는 항차 단위 AIS 시계열 데이터에서 신호가 누락된 구간의 상대좌표 `LAT_REL`, `LON_REL`을 복원하는 과제입니다.

AIS 데이터는 선박의 위치, 속도, 침로, 회전 정보 등을 포함하지만 실제 운항 환경에서는 악천후, 통신 음영, 장비 이상 등으로 인해 일부 구간이 누락될 수 있습니다. 이러한 누락은 항로 분석, ETA 예측, 위험 구역 회피 판단에 불확실성을 유발합니다.

본 프로젝트의 목표는 단순 보간을 넘어 물리적으로 자연스럽고 실제 항로 형태를 보존하는 복원 결과를 생성하는 것입니다.

## 2. 접근 전략
<img width="1384" height="542" alt="image" src="https://github.com/user-attachments/assets/f0d87221-064a-4a6c-90b3-417cd20b1bce" />
<br>

| 단계 | 내용 |
|---|---|
| 시간 정렬 | voyage 단위로 `UPDT_TM` 기준 정렬 후 `dt` 계산 |
| 기본 운항 피처 | `SOG`, `COG`, `ROT`, `DRFT`, `HD` 기반 방향·속도 피처 생성 |
| 물리 파생 피처 | 속도 벡터, 가속도, yaw rate, curvature, turn radius 생성 |
| 선박 크기 분류 | 선박 길이·폭 기반 volume proxy로 SMALL/MEDIUM/LARGE 구분 |
| Kalman + RTS | 상대좌표와 속도 의사관측을 활용해 물리적으로 안정적인 baseline 항로 생성 |
| Delta Modeling | 최근 시계열 window에서 다음 위치 변화량 `delta_lon`, `delta_lat`를 예측하는 구조 |
| Graph Temporal | 시간축 인접 그래프와 attention을 활용해 전역적인 궤적 형태 보정 |
| Block-wise Anchor | `IS_TARGET=1` 연속 구간을 block으로 보고 앞·뒤 관측 앵커 기준으로 복원 |
| Postprocess | Kalman baseline, anchor-scaled delta path, Gaussian center weight를 혼합해 최종 좌표 생성 |

## 3. 핵심 아이디어

### 3.1 Kalman Filter + RTS Smoother

`[lon, lat, vx, vy]` 상태공간을 사용해 항차별 기본 궤적을 생성합니다. 관측 좌표가 있는 구간에서는 위치와 속도 의사관측으로 보정하고, AIS 누락 구간에서는 constant velocity model로 상태를 예측합니다. Forward Kalman 이후 RTS backward smoothing을 적용해 전체 항차 관점에서 더 부드러운 baseline 경로를 만듭니다.

### 3.2 BiTCM Delta Model

Temporal Conv-Mixing 구조는 최근 `SEQ_LEN=20` 구간의 시계열 피처를 입력으로 받아 다음 위치 변화량 `delta_lon`, `delta_lat`를 예측하도록 설계했습니다. 이 구조는 국소적인 방향 변화, 감속, 급회전 패턴을 포착하는 데 초점을 둡니다.

### 3.3 Graph Temporal Refinement

각 시점을 노드로 보고 radius 내 인접 시점을 연결한 시간 그래프를 구성합니다. Graph convolution과 attention을 통해 BiTCM이 예측한 delta를 전역 시퀀스 관점에서 다시 보정합니다.

### 3.4 Block-wise Anchor Reconstruction

`IS_TARGET=1`이 연속된 구간을 하나의 복원 block으로 정의하고, block 앞뒤의 관측점을 anchor로 사용합니다. 모델 delta의 누적합이 anchor 간 벡터와 크게 어긋나는 경우 전체 delta를 scaling하여 anchor-to-anchor 이동량과 일관되게 맞춥니다. 이후 Kalman baseline과 alpha-blending하여 급격한 꺾임을 방지합니다.

## 4. 저장소 구조

<pre>
ais-route-reconstruction/
├─ configs/
│  └─ default.yaml
├─ src/
│  └─ route_reconstruction/
│     ├─ config.py
│     ├─ data.py
│     ├─ features.py
│     ├─ kalman.py
│     ├─ models.py
│     ├─ reconstruction.py
│     ├─ metrics.py
│     ├─ submission.py
│     └─ pipeline.py
├─ scripts/
│  └─ run_reconstruction.py
├─ tests/
├─ docs/
├─ reports/
├─ README.md
└─ requirements.txt
</pre>

## 5. 실행 방법

### 5.1 데이터 배치

대회 데이터는 저장소에 포함하지 않습니다. 아래 경로에 직접 배치합니다.

<pre>
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
</pre>

### 5.2 패키지 설치

<pre>
pip install -r requirements.txt
</pre>

### 5.3 전체 파이프라인 실행

<pre>
python scripts/run_reconstruction.py --config configs/default.yaml
</pre>

### 5.4 출력 파일

<pre>
outputs/
├─ submission.csv
├─ reconstructed_test.csv
└─ run_summary.json
</pre>

## 6. 설계 의도

이 프로젝트는 대회 제출 코드를 그대로 보관하는 대신, 실제 연구·실무형 항로 복원 파이프라인처럼 구성하는 것을 목표로 합니다.

- AIS 누락 구간 복원을 단순 보간이 아니라 물리 기반 시계열 복원 문제로 정의
- Kalman + RTS로 안정적인 baseline 항로 생성
- BiTCM/Graph Temporal 구조를 delta 보정 모듈로 분리
- target block 단위 anchor reconstruction으로 항로 형태 보존
- 속도, 곡률, gap 길이에 따라 Kalman과 모델 기반 경로를 동적으로 blending
