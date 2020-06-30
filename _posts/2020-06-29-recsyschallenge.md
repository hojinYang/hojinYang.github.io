---
title: "RecSys Challenge 참가 스토리"
categories: 
  - blogging
last_modified_at: 2020-06-22T13:00:00+09:00
author_profile: no
comments: true
---

추천시스템은 Applied ML 의 주요 분야로서 학계와 현업에서 활발한 연구가 진행되고 있다. 개인적으로 추천 분야에 흥미를 느끼고 데이터마이닝 랩에서 학부참여 연구를 하던 시절, 교수님의 소개로 RecSys Challenge에 대해 알게 되었다. 그 뒤로 두 번의 challenge에 참여해 보았는데, 이번 글에서는 대회 소개와 사용했던 모델, 그리고 참가 후기를 간단히 정리해 보았다.

![challenge logo](https://hojinYang.github.io/img/recsysc.png){:.aligncenter}

##  RecSys Challenge
ACM RecSys Challenge는 음악, SNS, 이커머스 등 다양한 도메인의 추천시스템 문제를 다루는 대회이며, 이 분야 최고 권위의 학회 중 하나인 [ACM RecSys](https://recsys.acm.org/)에서 매년 개최한다. 추천시스템 분야를 연구하는 전세계 대학뿐만 아니라 많은 기업이 참여해 자신들의 추천 모델을 검증해 보는데, 이전 대회 우승팀으로는 중국의 Alibaba(’16), 러시아의 Yandex('15), NVidia의 Rapids AI(’20)등이 있다. 높은 컴퓨팅파워와 데이터 파이프라인을 보유하고 있는 기업팀이 상위권을 차지하지만, 추천 분야를 연구하는 나와 같은 학생들도 대회를 통해 글로벌 Tech 기업의 데이터를 다뤄보며 현업의 문제들을 직접 해결해보는 기회를 얻을 수 있다.  

|<center>Year</center>|<center>Venue</center>|<center>Challenge Sponsor</center>|<center>Domain</center>|
|----|----------------------|-------|----------|
|2020|Rio de Janeiro, Brazil|Twitter|Tweet(SNS)|
|2019|Copenhagen, Denmark|Trivago|Hotel|
|2018|Vancouver, Canada|Spotify|Music|
|2017|Como, Italy|XING|Job|
|2013|Hong Kong|Yelp|Business|

 대회 진행 방식은 Kaggle과 유사하며 RecSys Challenge에서는 비교적 더 큰 데이터셋을 다루게 된다. 토론이 활발히 이루어지는 kaggle과는 달리 팀들간 교류가 거의 발생하지 않는다는 점도 특징이다. 대략 2월 말 주제와 데이터셋이 공개된 뒤 약 3개월간 대회가 진행되는데, 매년 가을 ACM RecSys와 함께 개최되는 Workshop에서 상위권 팀들의 솔루션을 확인할 수 있다. 개인적으로는 학부연구생을 하던 2018년, 인턴쉽을 하고 있는 2020년에 각각 팀장과 팀 멤버로 참가하였고 두 대회 모두 2위로 마무리하였다. 아래에 두 대회에 대한 설명과 팀이 사용했던 모델들을 간단히 정리해 보았다.

##  Spotify RecSys Challenge 2018: Automatic Playlist Continuation
### 개요

18년도 대회의 주제는 음악 스트리밍 서비스 [Spotify](https://www.spotify.com/us/)의 데이터를 활용해 재생목록에 어울리는 노래를 추천하는 Automatic Playlist Continuation(APC) System을 구현하는 것이었다. APC는 유저가 만든 플레이리스트를 분석해 그와 어울리는 곡들을 자동으로 계속해서 추가해 주는 추천 방식으로, 유저의 검색 비용을 줄이고 자연스레 앱에 오래 머무르도록 유도한다. 

이 대회에서는 수록된 곡의 일부(Seed track)가 플레이리스트 제목과 함께 주어졌을 때, 주어지지 않은 곡들을 예측하는 방식으로 모델을 평가하였다. Seed track은 0, 1, 5, 10, 25, 100개 등 다양한 크기로 주어졌으며 0의 경우 노래 없이 제목만으로 수록 곡들을 예측해야 했다(cold-start). 모델 학습을 위한 training set으로 Spotify의 유저들이 만든 플레이리스트 백만 개가 각 곡의 feature(앨범, 가수)와 함께 제공되었으며, 만 개의 플레이리스트가 test set으로 주어졌다. 성능 평가를 위한 metric으로 NDCG, R-precision등이 사용되었다. ~430 팀이 대회에 등록했으며, 더욱 자세한 정보는 [워크샵 홈페이지](http://www.recsyschallenge.com/2018/)와 [논문](https://arxiv.org/pdf/1810.01520.pdf)에서 확인할 수 있다. 

### 모델

우리 팀은 두 하위모델로 구성된 딥러닝 파이프라인을 활용하였는데, Collaborative Filtering(CF) 모델로 Autoencoder를 사용하였고, Character-level CNN을 바탕으로 플레이리스트 제목 기반 예측 모델을 구현하였다. 두 모델에서 얻어진 예측값을 블랜딩하여 최종 추천 곡 목록을 생성했다.  

- AutoRec

Playlist와 song을 각각 user와 item에 대응시켜 CF를 모델링하였다. 다만 이 대회에서는 training set에 등장하지 않은 플레이리스트로만 test set이 구성되어 있었기 때문에 user-side를 학습에 활용하는 U-I Matrix Factorization나 성능이 떨어지는 user-based CF는 사용하지 않았고, item 유사도 기반 추천 모델인 Item-based [AutoRec](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)을 베이스라인으로 사용하였다. 학습 단계는 우선 song list 중 플레이리스트에 포함된 노래를 bag-of-songs 벡터로 binary encoding한 뒤, encoder-decoder 구조를 활용해 원래의 곡 목록을 복원하는 autoencoder 방식으로 학습하였다 (Autoencoding 관점에서의 CF는 [이 영상](https://www.youtube.com/watch?v=yNVL7oBTjYE)에 잘 설명되어있다).

그러나 이 모델을 바로 학습에 사용할 수 없었다. 데이터셋에 등장하는 unique한 노래는 이백만 개가 넘었는데, 당시 사용하던 GTX 1080-ti box에 2M이 넘는 행/열을 갖는 encoder/decoder matrix를 올리는 것이 불가능했다. 데이터 크기를 효율적으로 줄이기 위해 전체 플레이리스트에서 5번 미만으로 등장하는 곡들을 제거하였는데, 확률적으로 이들이 정답일 가능성이 낮기 때문이다(1/2M~4/2M). 결과적으로 학습에 사용할 곡 개수를 원래의 1/4에 해당하는 오만 개로 줄일 수 있었다. 또한 곡을 제거하면서 발생하는 정보 손실을 최소화하기 위해 플레이리스트에 등장했던 가수 목록으로 bag-of-artists 벡터를 만들어 플레이리스트 벡터에 이어붙인 뒤 학습에 사용하였다. 

이번 대회에서 AutoRec 기반 모델을 사용한 다른 팀들은 중위권에 머물렀는데, 대회 task에 최적화된 학습 방법을 사용한 것이 우리팀의 상위권 입상에 주요했다. 매일 업데이트 되는 리더보드는 팀별로 test set 50%에 대한 평균 점수만 제한적으로 보여줬는데, local validation set을 통해 우리 베이스라인 모델이 input seed track 개수가 1개, 5개로 적은 경우 성능이 현저히 떨어진다는 문제점을 확인할 수 있었다. 대회에서는 다양한 seed track 개수 별 성능을 균등하게 고려하여 순위를 매겼기 때문에 이에 대한 개선이 필요했다. 이른바 input size에 robust한 AutoRec을 만들어야 했고, 이를 위해 사용한 두 가지 학습 방법으로 1) input중 일부를 batch마다 랜덤한 비율로 제거하였고(random zero-out ratio per batch), 2) input 크기가 일정하지 않기 때문에 모델 전체의 에너지를 일정하게 유지해주기 위해 input을 input size n으로 나눠줬다(1 -> 1/n, 이는 word2vec CBOW모델에서 [sum보다는 avg가 더 잘 작동하는 것](https://groups.google.com/forum/?fbclid=IwAR06M2_BW-qdTq0-9IuJMTUtqr6Z9UHNA5b3_VxPI-AeOe8YItIUYgiRW68#!topic/word2vec-toolkit/HlJyFACiVPE)과 비슷한 맥락이다). 이를 활용한 결과 input size가 작은 경우 뿐만 아니라 큰 경우 모두 모델의 성능이 향상되었다.  

- Char-level CNN

 대회가 진행되던 18년도는 CNN을 활용하여 NLP classification task를 해결하는 논문들이 등장하던 시기였고 우리도 이를 활용해 플레이리스트 제목을 처리하였다. 제목을 character-level로 쪼개어 각각을 k-dimentional character embedding으로 나타낸 뒤, 다시 이를 이어 붙여 title matrix를 만들었다. 이를 [Word-level CNN](https://arxiv.org/pdf/1408.5882.pdf)과 유사하게 다양한 필터 사이즈를 활용해 임베딩한뒤, output으로 해당 플레이리스트에 담긴 곡 목록을 복원하도록 학습하였다. 


#### 우승팀

18년 대회의 우승팀은 캐나다의 [Layer6 AI](https://layer6.ai/)이다. 1위팀과 3위팀 모두 우선 CF모델로 후보 음악을 추린 뒤 gradient boosting ensemble을 활용해 rerank하는 2-stage approach를 활용했다. 당시만 해도 GBM에 대해 잘 몰랐기도 했고 대부분의 작업을 혼자해야했기에 이를 시도해볼 시간적 여유가 없었다.  


##  Twitter RecSys challenge 2020: Tweet Engagement Prediction 
### 개요

20년도 대회의 주제는 [Twitter](https://twitter.com/home?lang=en)가 제공한 데이터를 활용해 tweet engagement(LIKE, RETWEET, REPLY, COMMENT) 예측 모델을 만드는 것이었다. 트위터 유저가 자신의 피드에 올라온 트윗들에 어떤 피드백을 주는지를 예측하는 모델로, 이를 기반으로 트위터는 개별 유저들의 입맞에 맞는 트윗이 우선적으로 노출되도록 피드를 재구성할 수 있다.  

이번 대회에서는 <트윗 작성자, 트윗, 유저>의 triplet이 주어졌을 때, 유저의 트윗 engagement를 예측하는 방식으로 모델을 평가하였다. 모델 학습을 위한 training set으로 일주일동안 발생한 약 이백만 개의 트윗 engagement 로그가 제공되었으며, 트윗 작성자와 유저 feature, 그리고 백만 개의 negative sample도 함께 제공되었다. Training 기간 이후 일주일간 새롭게 작성된 트윗들이 test set으로 주어졌으며, 네 가지 engagement들에 대한 각각의 PR-AUC와 cross-entropy loss로 모델의 정확도를 평가하였다. [1000+팀이 대회에 등록](https://twitter.com/trustswz/status/1275116571736891394)했으며, [워크샵 홈페이지](http://www.recsyschallenge.com/2020/)와 [논문](https://arxiv.org/pdf/2004.13715.pdf)에서 대회의 자세한 정보를 확인할 수 있다. 

### 모델

이번 대회에서 우리 팀이 사용한 딥러닝 파이프라인은 다음 세 종류의 입력을 받는다: 1)유저, 작성자, 트윗에서 추출한 features, 2) langugage model을 통해 학습한 트윗 embedding, 3) 유저와 작성자의 트윗 로그(history)를 반영한 user history embedding. 이 값들을 feed-forward network에 입력한 뒤 결과값으로 LIKE, REPLY, RETWEET, COMMENT 등 네가지 engagement의 확률을 계산하도록 학습하였다. 

- Feature Engineering

 XGBoost를 활용해 다양한 feature engineering/selection을 진행하고 여기서 얻어진 feature들을 DL 파이프라인에 활용하는 방식으로 진행하였다. XGB를 굳이 사용한 이유는 멀티스레드로 동작하는 자바 XGB 파이프라인이 이미 구현되어 있는 상태여서 빠른 속도로 feature들을 테스트 해볼 수 있었기 때문이다. 또한 ordinal, continuous, discrete등 다양한 종류의 feature가 주어졌을 때 정규화 등 추가적인 전처리를 필요로 하는 뉴럴 넷과는 달리 XGB는 이들을 바로 처리할 수 있어 받아 여러모로 시간을 줄여줬다. Feature만 사용했을 때 XGB의 성능은 뉴럴 넷과 거의 비슷하거나 조금 더 좋았는데, language model과 결합했을 때의 성능은 뉴럴 넷이 더 뛰어났다. 

- Language Models

영어, 일어, 한국어 등 다양한 언어로 작성된 트윗 데이터를 다루기 위해 transformer기반 multilingual language model인 [BERT-Base](https://huggingface.co/bert-base-multilingual-cased) 와 [XLM-Roberta-Large](https://huggingface.co/xlm-roberta-large)를 활용하였다. 이 pre-trained 모델들을 다시 트윗 데이터를 활용해 MLM Loss로 fine-tuning 하였는데, 이렇게 unsupervised 방식으로 학습한 트윗 임베딩을 feed-forward net의 입력값 중 하나로 사용하였다. BERT 혹은 XLM-R을 바로 engagement classification에 활용하는 end-to-end 모델 역시 시도하였는데, 성능은 조금 더 좋았지만 학습을 시작한지 얼마 안되어 overfitting이 발생한다는 애로사항이 있었다.

- User History Embedding

추천 모델에서 유저(아이템) 임베딩을 학습할 때는 주로 [Neural CF(NCF)](https://arxiv.org/pdf/1708.05031.pdf)) 방식을 사용한다. 그러나 이번 대회는 training/test set에 등장하는 negative sample이 모두 유저의 follower 그래프에서 추출되었고, 단순히 사용자간 거리를 임베딩하는 NCF 방식으로는 해당 트윗 engagement 여부를 판단하는데 충분한 정보를 주지 못했다. 그보다는 유저가 어떤 트윗 내용에 반응했는지가 target predcition에 더 중요한 요소로 판단하였다. target tweet을 기준으로 유저가 engage한 트윗들의 attention score를 계산하였고 이를 통해 얻은 user history embedding을 feed-forward의 입력 중 하나로 사용하였다.

#### 우승팀

20년 대회의 우승팀은 NVidia의 [Rapids AI](https://rapids.ai/)이다. GPU기반의 data science open source를 개발하는 팀으로, kaggle former no.1인 giba를 비롯 다수의 grandmaster가 팀을 이뤘다. 18년도에는 아쉽게 2위를 차지했다면 20년도 대회는 1위팀과의 격차가 꽤 벌어졌는데, 딥러닝 없이 gradient boosting기반의 모델을 사용한 것으로 보인다. Feature engineering과정에서 target encoding과 frequency encoding을 사용했던 게 주요했다고 [밝혔다](https://twitter.com/tunguz/status/1275116189551865862).    

### 마치며

두 대회에서 사용한 코드는 [github](https://github.com/hojinYang)에서 확인할 수 있다. 몇 가지 느낀점을 정리하며 이번 포스팅을 마친다. 

- 모델보다는 데이터 파이프라인

앞서 두 대회에서 사용한 모델들을 간략히 소개하였지만 사실 이런 대회에서는 다양한 모델/feature들을 실험하기 좋게 짜놓은 깔끔한 데이터 파이프라인이 순위를 결정한다고 생각한다. 대략 대회에 집중할 수 있는 시간이 2개월이 남짓인데, 앞단의 파이프라인을 잘 구축해 놓아야 이후 다양한 실험을 진행하는 데 소요되는 시간을 줄일 수 있다. 이 부분은 이론보다는 대회 참여를 통해 다양한 종류의 데이터를 다뤄보면서 경험을 누적 시켜야 할 것 같다. 

- 빠른 구현과 검증

주어진 문제를 해결하는 데 정해진 정답이 있는 것이 아니기 때문에 최대한 다양한 가설을 새워보고 이를 코드를 돌려서 확인해봐야 한다. 빠른 실험을 위해서는 앞서 말한 유지보수가 쉬운 파이프라인을 짜는 것이 중요하다. 또한 RecSys와 같이 큰 데이터를 다룰 경우에는 전체 데이터를 가지고 실험하는 것이 하루~이틀은 걸릴 수 있기 때문에 자신의 로컬 머신에 subsampling한 training set과 validation set 환경을 구축해 시간을 아끼는 것이 필요하다. 

- Gradient Boosting Ensemble, 그리고 딥러닝

모델 측면에서 보았을 때는 GBM을 사용하는 것이 대세이고 성능도 잘 나온다. 대부분의 data competition에서 주어지는 tabular data에 가장 잘 맞는 모델이기도 하고, 앞서 말했듯 ordinal, continuous, discrete feature에 따른 scaling issue도 따로 없기 때문에 feature engineering에만 집중할 수 있다는 장점이 있다. 다만 NLP나 CV등 비정형 데이터는 다루는 문제는 딥러닝을 활용하는 것이 일반적인데, 20년 대회와 같이 tabular 데이터와 비정형 데이터가 같이 주어지는 경우 두 모델을 어떻게 결합해야 좋을지에 대한 고민을 하게 되었다(2-stage로 가야할지 아니면 feature input을 MLP를 활용해 DL로 처리해야 하는지). [TabNet](https://arxiv.org/pdf/1908.07442.pdf)등 tabular data를 다루는 네트워크나 아니면 아예 AutoML기반 솔루션([아마존](https://arxiv.org/pdf/2003.06505.pdf), [구글](https://ai.googleblog.com/2019/05/an-end-to-end-automl-solution-for.html))도 활용하는 것 같은데, 기회가 되면 공부해봐야겠다. 혹시 이와 관련한 좋은 레퍼런스가 있으면 소개해주면 큰 도움이 될 것 같다! :)  