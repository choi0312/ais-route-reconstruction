# Modeling Strategy

## 1. Problem Formulation

본 문제는 항차 단위 AIS 시계열에서 누락된 `LAT_REL`, `LON_REL` 좌표를 복원하는 trajectory reconstruction 문제이다. 단기 보간뿐 아니라 수십 분에서 수시간 단위의 긴 gap을 복원해야 하므로, 위치 정확도와 경로 형태 보존을 함께 고려해야 한다.

## 2. Hybrid Design

전체 구조는 물리 모델과 데이터 기반 모델을 결합한다.

- Kalman Filter + RTS Smoother: 물리적으로 안정적인 baseline trajectory 생성
- BiTCM: 최근 window 기반 local temporal pattern에서 delta 예측
- Graph Temporal Layer: 전체 sequence curvature와 장단기 의존성 보정
- Block-wise Anchor Reconstruction: 관측 anchor 사이에서 target block을 안정적으로 복원

## 3. Feature Engineering

`UPDT_TM` 기반 `dt`, `COG`의 sin/cos 변환, `SOG` 기반 속도 벡터, 가속도, yaw rate, curvature, turn radius, vessel size class를 생성한다.

## 4. Reconstruction Logic

`IS_TARGET=1` 연속 구간을 block으로 분리하고, 앞뒤 관측 anchor를 기준으로 delta path를 scaling한다. 이후 block 내부 위치에 따라 Gaussian weight를 부여하고 Kalman baseline과 alpha-blending한다.
