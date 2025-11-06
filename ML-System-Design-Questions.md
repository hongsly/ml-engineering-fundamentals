# ML System Design Interview Questions

This document contains ML system design questions commonly asked in ML Engineer interviews at top tech companies.

---

## How to Approach ML System Design Questions

### The Framework (Use for every question)

**1. Clarify Requirements (5-10 minutes)**
- Understand the business objective
- Define success metrics
- Clarify scale (users, requests, data volume)
- Understand latency and accuracy requirements
- Ask about existing infrastructure

**2. Define the ML Problem (5 minutes)**
- Frame as ML task (classification, ranking, recommendation, etc.)
- Define features (input) and labels (output)
- Discuss data sources and availability

**3. Data Pipeline (10 minutes)**
- Data collection and storage
- Data labeling strategy
- Feature engineering
- Train/validation/test split
- Data quality and monitoring

**4. Model Development (10 minutes)**
- Baseline approach
- Model selection (start simple, then complex)
- Training strategy
- Evaluation metrics
- Offline evaluation approach

**5. Deployment & Serving (10 minutes)**
- Model serving architecture
- Latency requirements and optimization
- A/B testing and gradual rollout
- Model versioning

**6. Monitoring & Maintenance (5 minutes)**
- Performance monitoring
- Data drift detection
- Model retraining strategy
- Feedback loop

**7. Discussion & Trade-offs (remaining time)**
- Scalability concerns
- Cost considerations
- Edge cases and failure modes
- Privacy and ethics

---

## Question 1: Design a Video Recommendation System

### Problem Statement
Design a machine learning system to recommend videos to users on a video streaming platform with 500M users and 1B videos.

### Key Requirements to Clarify
- What's the primary goal? (Watch time, engagement, user satisfaction)
- Real-time or batch recommendations?
- Cold start problem for new users/videos?
- Latency requirements? (<100ms?)
- Personalization level?

### Solution Outline

**1. ML Problem Framing**
- **Type**: Ranking/Recommendation problem
- **Input**: User features, video features, context
- **Output**: Ranked list of N videos (e.g., top 100)
- **Objective**: Maximize watch time or engagement

**2. Data Sources**
- User interactions (views, likes, searches, watch time)
- Video metadata (title, description, tags, category)
- User profile (demographics, preferences, history)
- Context (time of day, device, location)

**3. System Architecture**

```
User Request → Candidate Generation → Ranking → Re-ranking → Serve
   ↓                  ↓                  ↓           ↓
Context          (100k → 1k)      (1k → 100)   Business Rules
```

**Stage 1: Candidate Generation (Recall)**
- Collaborative filtering (user-user, item-item)
- Content-based filtering (video features)
- Trending/popular videos
- Goal: Reduce 1B videos to ~100k candidates quickly

**Stage 2: Ranking Model (Precision)**
- Deep neural network (e.g., two-tower model)
- Features:
  - User: watch history embeddings, demographics, preferences
  - Video: embeddings from title/thumbnail, engagement stats
  - Context: time, device, previous session behavior
  - Cross features: user-video interaction history
- Objective: Predict engagement score (e.g., watch time)
- Reduce 100k → 1k videos

**Stage 3: Re-ranking**
- Diversity considerations (avoid all same genre)
- Business rules (exclude watched, apply filters)
- Freshness (boost recent uploads)
- Final 100 videos

**4. Training Data**
- Positive examples: Videos watched >30s
- Negative examples: Impressions but not clicked, or clicked but abandoned
- Handle implicit feedback (no explicit ratings)

**5. Features**
- **User features**:
  - Watch history embeddings (learned)
  - Engagement rate, avg watch time
  - Demographics, language preference
- **Video features**:
  - Video embeddings (from title, description, thumbnail)
  - Upload date, category, creator
  - Historical engagement (CTR, watch time, likes)
- **Context features**:
  - Time of day, day of week
  - Device type
  - Search query (if from search)

**6. Model Architecture**
- Two-tower neural network:
  - User tower: Encodes user + context → user embedding
  - Video tower: Encodes video features → video embedding
  - Score = dot product of embeddings
- Alternative: Wide & Deep, DeepFM

**7. Training Strategy**
- Offline training on historical data (past 30-90 days)
- Negative sampling (sample non-clicked videos)
- Handle class imbalance (very few videos watched vs. impressed)
- Regular retraining (daily or weekly)

**8. Evaluation Metrics**
- Offline: AUC, precision@k, recall@k, NDCG
- Online: CTR, watch time, user engagement, session length
- A/B testing for online evaluation

**9. Serving Architecture**
- Candidate generation: Pre-compute and cache (batch processing)
- Ranking: Real-time scoring with low latency (<100ms)
- Use approximate nearest neighbor (ANN) for fast retrieval
- Distributed serving with load balancing

**10. Challenges & Solutions**

| Challenge | Solution |
|-----------|----------|
| Cold start (new users) | Use demographic-based defaults, popular videos, onboarding quiz |
| Cold start (new videos) | Use content features, boost in ranking temporarily |
| Scalability (1B videos) | Two-stage pipeline (candidate gen + ranking) |
| Real-time updates | Stream processing for recent interactions |
| Filter bubbles | Diversity in re-ranking, exploration vs. exploitation |
| Latency | Caching, approximate search, model compression |

**11. Monitoring**
- Model performance: CTR, watch time (by cohort)
- Data drift: Feature distribution changes
- System health: Latency, errors, cache hit rate
- Business metrics: User engagement, retention

**12. Iteration & Improvement**
- Incorporate new signals (e.g., social sharing)
- Experiment with model architectures
- Personalize diversity/exploration parameters
- Multi-objective optimization (watch time + satisfaction)

---

## Question 2: Design a Search Ranking System

### Problem Statement
Design a machine learning system to rank search results for an e-commerce platform like Amazon or a search engine like Google.

### Key Concepts to Cover

**ML Problem**: Learning to Rank (LTR)
- Pointwise: Predict relevance score for each document
- Pairwise: Learn relative ordering of documents
- Listwise: Directly optimize ranking metric (NDCG)

**Features**:
- Query features: Length, type (navigational/informational), user intent
- Document features: Title, content, metadata, popularity
- Query-document features: TF-IDF, BM25, semantic similarity
- User features: Location, past behavior, preferences
- Context: Time, device

**Candidate Generation**:
- Traditional IR: Inverted index, BM25
- Semantic search: Query/document embeddings, ANN search

**Ranking Model**:
- LambdaMART, LambdaRank (gradient boosting)
- Neural ranking models (BERT for text matching)
- Pointwise: Predict click probability
- Pairwise: RankNet, learn doc_i > doc_j

**Evaluation**:
- Offline: NDCG, MRR (Mean Reciprocal Rank), MAP (Mean Average Precision)
- Online: CTR, dwell time, conversion rate

**Key Challenges**:
- Position bias: Users click top results more
- Long-tail queries: Few training examples
- Freshness: New documents need to be indexed quickly
- Personalization vs. consistency

---

## Question 3: Design a Fraud Detection System

### Problem Statement
Design a system to detect fraudulent transactions in real-time for a payment platform (e.g., credit card fraud, payment fraud).

### Solution Outline

**1. ML Problem**
- Binary classification: Fraud (1) vs. Legitimate (0)
- Extreme class imbalance (fraud is rare: 0.1-1%)
- Need real-time prediction (<100ms)

**2. Features**
- **Transaction features**: Amount, merchant, location, time
- **User features**: Account age, transaction history, velocity
- **Aggregated features**:
  - Number of transactions in last hour/day
  - Average transaction amount (rolling window)
  - Number of unique merchants in last day
- **Derived features**:
  - Distance from last transaction
  - Time since last transaction
  - Deviation from user's normal behavior

**3. Data Pipeline**
- Real-time stream processing (Kafka, Flink)
- Feature store for fast feature retrieval
- Historical data for training (labeled fraud cases)

**4. Model Selection**
- Start simple: Logistic regression with engineered features
- Gradient boosting (XGBoost, LightGBM) - works well with tabular data
- Neural network with embeddings (for categorical features)
- Ensemble of models

**5. Handling Class Imbalance**
- Oversampling minority class (SMOTE)
- Undersampling majority class
- Use class weights in loss function
- Use appropriate metrics (precision, recall, F1, not accuracy)

**6. Evaluation Metrics**
- Precision: Of flagged transactions, how many are truly fraud?
- Recall: Of all fraud cases, how many did we catch?
- F1-score or F-beta (tune beta based on cost of false positives vs. false negatives)
- ROC-AUC, Precision-Recall AUC

**7. Real-time Serving**
- Pre-compute user features and cache
- Feature store for fast lookup
- Model serving with <100ms latency
- Rule-based filters for obvious cases

**8. Human-in-the-Loop**
- High-confidence predictions: Auto-block (fraud) or approve (legit)
- Medium confidence: Route to manual review
- Collect feedback to improve model

**9. Challenges**

| Challenge | Solution |
|-----------|----------|
| Extreme imbalance | Resampling, class weights, appropriate metrics |
| Evolving fraud patterns | Continuous retraining, monitor drift |
| Latency | Feature caching, fast models, rule-based shortcuts |
| False positives cost | Tune threshold, use two-stage (rules + ML) |
| Adversarial nature | Ensemble models, anomaly detection, regular updates |

**10. Monitoring**
- Model: Precision, recall, false positive rate (by threshold)
- Data drift: Feature distributions over time
- Fraud patterns: New attack vectors
- Business: $ saved, user friction, review queue size

---

## Question 4: Design a Feed Ranking System (Facebook/LinkedIn/Twitter)

### Problem Statement
Design a system to rank posts in a social media feed to maximize user engagement.

### Key Points

**ML Problem**: Ranking/Personalization
- Multi-objective: Likes, comments, shares, dwell time
- Balancing objectives (engagement vs. quality)

**Candidate Generation**:
- Friends' posts, followed accounts, sponsored content
- Time-based filtering (recent posts)
- Reduce millions of posts to 1000s

**Ranking**:
- Predict engagement probability for each post
- Features: User, post, user-post interaction history, social graph
- Multi-task learning (predict like, comment, share separately)

**Challenges**:
- Cold start: New users, new creators
- Diversity: Avoid showing only one type of content
- Echo chambers: Balance engagement with content diversity
- Viral content: Detect and boost (or suppress) going-viral posts
- Freshness: Recent posts should be prioritized

**Architecture**:
```
All Posts → Candidate Gen (10k) → Ranking (500) → Re-rank → Serve (50-100)
```

**Evaluation**:
- Offline: Multi-task loss, per-task AUC
- Online: Engagement rate, session time, DAU retention

---

## Question 5: Design an Ad Click Prediction System

### Problem Statement
Design a system to predict the probability a user will click on an ad (CTR prediction).

### Key Concepts

**ML Problem**: Binary classification (click or no click)
- Optimize for CTR (click-through rate)
- Ads are ranked by expected value: bid × P(click)

**Features**:
- User: Demographics, browsing history, past ad interactions
- Ad: Advertiser, creative, category
- Context: Page content, time, device
- Cross features: User-ad similarity

**Model Architecture**:
- Logistic regression with feature engineering (baseline)
- Factorization machines (handle sparse features)
- Deep learning: Wide & Deep, DeepFM, DCN (Deep & Cross Network)
- Learn embeddings for categorical features

**Training**:
- Positive: Clicked ads
- Negative: Impressed but not clicked
- Handle imbalance (CTR typically 0.5-2%)

**Calibration**:
- Predicted probability should match actual click rate
- Use calibration techniques (Platt scaling, isotonic regression)

**Evaluation**:
- Offline: AUC, log loss, calibration
- Online: CTR, revenue per impression

**Challenges**:
- Selection bias: Only see impressions from current model
- Delayed feedback: Clicks may happen minutes later
- Adversarial advertisers: Click fraud, misleading ads

---

## Question 6: Design a Content Moderation System

### Problem Statement
Design a system to automatically detect and remove harmful content (hate speech, spam, violence) on a social platform.

### Solution Outline

**1. ML Problem**
- Multi-class classification or multi-label classification
- Classes: Hate speech, spam, violence, nudity, etc.
- Highly imbalanced (most content is benign)

**2. Data**
- Text, images, video content
- User reports and moderator labels
- Contextual information (user history, comments)

**3. Model Architecture**
- **Text**: Fine-tuned BERT or RoBERTa
- **Images**: CNN (ResNet, EfficientNet) or Vision Transformer
- **Video**: Frame sampling + temporal model
- **Multimodal**: Combine text + image + video

**4. Pipeline**
```
Content Upload → Preprocessing → Model Inference → Confidence Score
                                                          ↓
                              Auto-remove (high) | Human Review (medium) | Approve (low)
```

**5. Human-in-the-Loop**
- High confidence (>0.95): Auto-remove
- Medium (0.5-0.95): Queue for human review
- Low (<0.5): Approve
- Collect moderator decisions to improve model

**6. Evaluation**
- Precision: Of flagged content, how much is truly harmful?
- Recall: Of all harmful content, how much did we catch?
- Trade-off: False positives upset users, false negatives damage platform
- Online metrics: User reports, moderator review time

**7. Challenges**

| Challenge | Solution |
|-----------|----------|
| Evolving content (new slang, memes) | Continuous retraining, community input |
| Context-dependent (sarcasm, quotes) | Multimodal models, consider context |
| Adversarial users | Ensemble models, rule-based + ML |
| Scale (millions of posts/day) | Efficient models, prioritize risky content |
| Cultural/language differences | Region-specific models, local moderators |

**8. Ethical Considerations**
- Bias: Avoid over-flagging certain groups
- Transparency: Explain why content was removed
- Appeals: Allow users to contest decisions
- Privacy: Handle sensitive content appropriately

---

## Question 7: Design a Real-time Bidding System for Ads

### Problem Statement
Design an ML system for real-time bidding (RTB) in ad auctions, where advertisers bid for ad impressions in <100ms.

### Key Concepts

**Problem**: Predict value of an impression and bid accordingly
- Value = P(click) × P(conversion | click) × conversion_value
- Bid to maximize ROI while staying within budget

**Components**:
1. **CTR Prediction**: Probability user clicks on ad
2. **CVR Prediction** (Conversion Rate): Probability user converts after click
3. **Bid Optimization**: How much to bid given predicted value

**Challenges**:
- Ultra-low latency (<100ms including model inference)
- Data sparsity (specific user-ad combinations rarely seen)
- Budget pacing (don't spend budget too fast or too slow)
- Selection bias (only observe outcomes for won auctions)

**Architecture**:
- Lightweight models (logistic regression, small NNs)
- Feature caching and pre-computation
- Approximate inference for speed
- Distributed serving

**Evaluation**:
- Offline: AUC for CTR/CVR
- Online: Cost per click (CPC), cost per acquisition (CPA), ROI

---

## Question 8: Design a News Article Recommendation System

### Problem Statement
Design a system to recommend news articles to users on a news platform.

### Key Differences from Video Recommendation

**Unique Challenges**:
- **Freshness**: News becomes stale quickly (within hours)
- **Cold start for items**: New articles have no engagement history
- **Diversity**: Users want varied perspectives and topics
- **Filter bubbles**: Don't create echo chambers
- **Breaking news**: Detect and surface important breaking news

**Solution Approach**:

**1. Candidate Generation**
- **Trending**: Recent articles with high engagement velocity
- **Personalized**: Based on user's reading history (topics, sources)
- **Diversified**: Ensure variety of topics and perspectives
- **Breaking news detector**: Identify sudden spikes in coverage

**2. Ranking**
- Features:
  - Article: Title, topic, source, recency, engagement (clicks, time spent)
  - User: Reading history, topics of interest, demographic
  - Context: Time of day, device
- Model: Gradient boosting or neural network
- Multi-objective: CTR + time spent + diversity

**3. Recency Handling**
- Exponential decay on article age
- Boost new articles temporarily (exploration)
- Use content features (not just collaborative filtering)

**4. Evaluation**
- CTR, time spent per article
- Session length, return rate
- Topic diversity in recommendations

---

## Additional Practice Problems

Here are more problems to practice (outline solutions yourself):

1. **Design a spam email detection system**
2. **Design a music recommendation system (Spotify)**
3. **Design a ride demand prediction system (Uber/Lyft)**
4. **Design an image similarity search system**
5. **Design a document ranking system for search**
6. **Design a chatbot intent classification system**
7. **Design a product recommendation system (e-commerce)**
8. **Design a dynamic pricing system**
9. **Design a language translation system**
10. **Design a voice assistant (speech recognition + NLU)**

---

## Common Themes Across All Problems

### Data
- Always discuss data collection, labeling, and quality
- Training/validation/test split strategy
- Handling class imbalance

### Models
- Start simple (baseline), then add complexity
- Justify model choice based on problem
- Discuss trade-offs (accuracy vs. latency vs. interpretability)

### Evaluation
- Define offline metrics (AUC, precision, recall, NDCG, etc.)
- Define online metrics (user engagement, business metrics)
- A/B testing methodology

### Deployment
- Serving architecture (real-time vs. batch)
- Latency requirements and optimization
- Model versioning and rollback

### Monitoring
- Performance monitoring (model + system)
- Data drift and concept drift
- Retraining strategy

### Scalability
- Handle large data volume
- Handle high traffic (requests per second)
- Distributed training and serving

---

## Tips for Acing System Design Interviews

1. **Clarify requirements upfront** - Don't make assumptions
2. **Use the framework** - Cover all components systematically
3. **Think out loud** - Explain your reasoning
4. **Draw diagrams** - Visualize architecture and data flow
5. **Discuss trade-offs** - No perfect solution, everything has trade-offs
6. **Be concrete** - Give specific examples (e.g., "XGBoost for ranking")
7. **Consider scale** - Think about 100M users, 1B items
8. **Mention monitoring** - Production ML requires monitoring
9. **Ask questions** - Engage with interviewer, it's a discussion
10. **Practice, practice, practice** - Do 10+ problems before interview

---

## Resources

- "Machine Learning System Design Interview" by Ali Aminian & Alex Xu
- "Designing Machine Learning Systems" by Chip Huyen
- ML System Design practice: https://www.educative.io/courses/machine-learning-system-design
- Real-world ML architectures: Netflix, Google, Facebook engineering blogs

---

**Next Steps**: Practice designing solutions for all 8 main questions + 10 additional problems. Time yourself (45 minutes per problem). Record yourself and review.
