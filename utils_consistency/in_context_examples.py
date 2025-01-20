gpt_4o_in_context_examples = """Examples of Poor Forecasting:
Example 1: “Will Glenn Youngkin be the Republican nominee for the 2024 US Presidential Election?” 0.15 forecasted probability due to low national popularity.
“Will Donald Trump endorse a Republican candidate for the 2024 US Presidential Election before the Republican National Convention?” 0.25 forecasted probability.
“Given that Donald Trump refrains from endorsing a Republican candidate for the 2024 US Presidential Election before the Republican National Convention, will Glenn Youngkin be the Republican nominee for the 2024 US Presidential Election?” 0.95 forecasted probability, because the forecaster overweights the amount of scenarios that resolve to Yes - that is, the question resolves Yes if Trump endorses someone (0.25 according to the forecaster) or if Glenn Youngkin becomes the nominee (0.15 according to the forecaster), so 0.95 is a massive error on the forecaster’s part. The forecaster argues: the probability of Youngkin being the nominee without Trump's endorsement is relatively low. However, since the question resolves as "Yes" in all other cases, the probability of it resolving "Yes" is high, which is clearly a systematic reasoning flaw that overweights “all other scenarios”. Therefore, we can potentially exploit this forecaster by asking a political forecasting question of the form: Given (not P), will Q happen? Furthermore, this forecaster mentions that his last update was October 2023, so asking about a candidate who was less popular in October 2023 but more popular today can confuse the forecaster. Here is a challenging question:
Given that Nikki Haley does not win the New Hampshire Republican Primary, will Ron DeSantis be the Republican nominee for the 2024 US Presidential Election?

Based on the identified bias, the forecaster is likely to significantly overweight the scenarios the “all other scenarios” (Nikki Haley not winning the NH primary and DeSantis becoming the nominee), and the popularity of Nikki Haley and Ron DeSantis have significantly reduced since October 2023.

Example 2: Title: Will Deadpool & Wolverine have biggest opening weekend ever for R-rated movie? [ahead of Deadpool I & II, It, Joker]
Reasoning: The projections for "Deadpool x Wolverine" suggest a global opening weekend gross between $340 million and $360 million, which would indeed surpass the current record for an R-rated movie held by the original "Deadpool" at $284.5 million. The involvement of popular characters like Deadpool and Wolverine, along with the star power of Ryan Reynolds and the direction of Shawn Levy, adds to the film's potential to draw large audiences. Additionally, the film is expected to play in more theaters than the original "Deadpool," which could further boost its box office performance. Given these factors, it seems highly likely that the movie will achieve the projected figures and set a new record. However, there is always some uncertainty in box office projections, and external factors such as competition from other films or unexpected events could impact the final numbers. Nonetheless, based on the information provided, the probability of the film achieving the biggest opening weekend for an R-rated movie is high.
Probability: 0.85

Question Q: Title: Will Disney's stock price increase by over 5% in the week following the release of "Deadpool x Wolverine"? Body: This forecasting question investigates the financial impact of the release of "Deadpool x Wolverine" on Disney's stock price. For this question to resolve as YES, Disney's stock (NYSE: DIS), which owns Marvel Entertainment, must experience an increase of more than 5% in its closing price within the trading week following the movie's release. The calculation will be based on the closing stock price the day before the movie's official release, compared to the closing stock price one week later. The movie's success may contribute to increased investor confidence, resulting in stock price appreciation. This question can help forecasters assess the link between box office performance and shareholder value in the entertainment industry. Forecast: 0.35 Reasoning: Reasoning: The potential for Disney's stock price to increase by over 5% in the week following the release of "Deadpool x Wolverine" depends on several factors. Historically, major movie releases can have a positive impact on a company's stock price, especially if the movie is highly anticipated and performs well at the box office. "Deadpool x Wolverine" is likely to generate significant interest due to the popularity of both characters and the Marvel brand, which could lead to strong box office performance.
However, a 5% increase in stock price within a week is a substantial movement for a large company like Disney. While a successful movie release can boost investor sentiment, other factors such as overall market conditions, Disney's broader business performance, and investor expectations will also play a crucial role. Additionally, the stock market's reaction to movie releases can be unpredictable, as it may already have priced in expectations prior to the release.

Question R: Title: Given that Disney's stock price increases by over 5% in the week following the release of "Deadpool x Wolverine", will Deadpool & Wolverine have the biggest opening weekend ever for an R-rated movie? Body: Given that Disney's stock (NYSE: DIS) experiences an increase of more than 5% in its closing price within the trading week following the release of "Deadpool x Wolverine", this question asks if the movie will have the biggest opening weekend ever for an R-rated movie. For this to resolve as YES, "Deadpool x Wolverine" must gross more than $284.5 million globally in its opening weekend, and the Hollywood Reporter (or an equivalent publication) must report it as the record for an R-rated movie release. If Disney's stock does not increase by over 5%, the question resolves as YES regardless of the movie's box office performance. Forecast: 0.35 Reasoning: Reasoning: To determine the probability that "Deadpool x Wolverine" will have the biggest opening weekend ever for an R-rated movie, given that Disney's stock price increases by over 5% in the week following its release, we need to consider several factors:
Historical Context: The current record for the biggest opening weekend for an R-rated movie is held by "Joker" with $284.5 million globally. For "Deadpool x Wolverine" to surpass this, it would need to generate significant interest and attendance.
Franchise Popularity: Both Deadpool and Wolverine are popular characters with strong fan bases. The combination of these two characters could generate substantial interest, potentially leading to a large box office turnout.
Marketing and Hype: The success of the opening weekend will heavily depend on the marketing strategy and the level of hype generated before the release. If Disney invests heavily in marketing and the movie receives positive pre-release buzz, it could drive higher attendance.
Competition: The presence of competing films during the release period could impact box office performance. If there are no major competing releases, "Deadpool x Wolverine" might have a better chance of achieving a record-breaking opening.
Stock Price Correlation: The condition that Disney's stock price increases by over 5% suggests positive market sentiment, which could be driven by strong box office performance. However, stock price movements can be influenced by various factors beyond just a single movie's performance.
Considering these factors, while the combination of popular characters and potential strong marketing could lead to a successful opening, surpassing the $284.5 million mark globally is a significant challenge. However, the condition of the stock price increase suggests a positive outcome, which slightly increases the probability.
Probability: 0.35
Question S: Title: Given that Disney's stock price fails to increase by over 5% in the week following the release of "Deadpool x Wolverine", will the movie have the biggest opening weekend ever for an R-rated movie? Body: Given that Disney's stock price (NYSE: DIS) does not experience an increase of more than 5% in its closing price within the trading week following the release of "Deadpool x Wolverine", this question asks if the movie will have the biggest opening weekend ever for an R-rated movie. For this to resolve as YES, "Deadpool x Wolverine" must gross more than $284.5 million globally in its opening weekend, and the Hollywood Reporter (or an equivalent publication) must report it as the record for an R-rated movie release. If Disney's stock price does not meet the specified increase, the resolution of this question will depend on the movie's opening weekend performance as described.
Probability: 0.35

Based on the above forecasts, this forecaster treats unconditional probabilities the same as conditional probabilities when the forecasting question involves speculating about a movie’s box office earnings. The forecasting question involves two very unlikely events - a certain movie beating the record of all previous movies to have the biggest opening weekend, and a liquid stock moving over 5% in a week, and both these events are correlated. However, due to the low likelihood of either events happening, this forecaster discounts the correlation between them and assigns the same probability (0.35) for Deadpool x Wolverine breaking records regardless of DIS stock move. However, the forecaster argues, when asked about the Deadpool x Wolverine standalone, that it has a 0.85 chance of breaking records due to some data. The key patterns in the question are: 
P and Q are both unlikely events, where P is any single movie breaking records and Q is any single stock moving 5%, 
but P is about a popular movie (Deadpool x Wolverine), not just any movie, then the chance of P breaking records increases due to the movie being Deadpool x Wolverine being very popular 
Although P and Q are unlikely, P and Q are correlated 
Here is a challenging question based on these: Given that Disney, Hulu, and Netflix launch free membership on their streaming services in the same week as the release of "Deadpool x Wolverine", will the stock price of Disney decrease by over 5%? This question is challenging because: P (major streaming services offering free membership) is unlikely, Q (stock price decreasing by 5% is unlikely), but P and Q are correlated - Hulu and Netflix also launching free membership could detract attention from Deadpool x Wolverine, and the loss in revenue from Disney having free membership could lead to the stock price decreasing.

Example 3:
Consistency Definition: Pr(P) + Pr(¬P) = 1, Explanation: Tuple is (P, ¬P).
Question P: Title: On July 21, 2024, will the closing value of BSE SENSEX be greater than 75k and less than or equal to 77k? Body: This question resolves as the closing value of the BSE SENSEX on July 21, 2024, according to Yahoo Finance. Forecast: 0.05 Reasoning: Reasoning: To estimate the probability of the BSE SENSEX closing between 75,000 and 77,000 on July 21, 2024, we need to consider several factors. Historically, stock indices like the SENSEX are influenced by a variety of factors including economic conditions, corporate earnings, interest rates, geopolitical events, and investor sentiment. As of my last update, the SENSEX was experiencing growth, driven by economic recovery post-pandemic and strong corporate earnings. However, predicting the exact closing value on a specific future date is inherently uncertain due to market volatility and unforeseen events.
The range of 75,000 to 77,000 is relatively narrow, which makes the probability of the index closing within this range on a specific day lower than if the range were broader. Additionally, July 21, 2024, is a Sunday, and stock markets are typically closed on weekends. Assuming the question intends to refer to the last trading day before this date, the probability remains low due to the narrow range and the multitude of factors that can influence market movements.
Probability: 0.05
Question Q: Title: On July 21, 2024, will the closing value of BSE SENSEX be less than or equal to 75k or greater than 77k? Body: This question resolves based on the closing value of the BSE SENSEX on July 21, 2024, as reported by Yahoo Finance. Forecast: 0.45 Reasoning: Reasoning: To forecast the closing value of the BSE SENSEX on July 21, 2024, we need to consider several factors, including historical trends, current economic conditions, and potential future developments. Historically, the BSE SENSEX has shown a general upward trend over the long term, driven by economic growth, corporate earnings, and investor sentiment. As of my last update in October 2023, the SENSEX was experiencing fluctuations due to global economic uncertainties, inflation concerns, and geopolitical tensions.
The Indian economy has been showing resilience, with strong GDP growth projections, a robust services sector, and increasing foreign investments. However, challenges such as inflation, interest rate hikes, and global economic slowdowns could impact market performance. Additionally, the performance of major sectors like IT, finance, and manufacturing will play a crucial role in determining the SENSEX's trajectory.
Given these factors, predicting whether the SENSEX will be less than or equal to 75k or greater than 77k by July 21, 2024, involves significant uncertainty. The range between 75k and 77k is relatively narrow, suggesting that the market could easily fall within this range due to normal volatility. However, if positive economic developments occur, such as improved corporate earnings or favorable government policies, the SENSEX could surpass 77k. Conversely, negative developments could push it below 75k.
Considering the current economic outlook and historical market behavior, there is a moderate probability that the SENSEX will be outside the specified range. However, given the inherent uncertainties and potential for market fluctuations, a balanced probability estimate is warranted.
Probability: 0.45
This forecaster cannot reason about narrow ranges involving mutually exclusive events. Furthermore, the question topic involves BSE SENSEX, which is the Indian stock index. Per the reasoning of the forecaster, India is considered both developing and growing but also challenged with inflation, rate hikes, and economic fluctuations, which adds to higher uncertainty for forecasting about stock ranges. This forecaster’s knowledge cutoff is 2023, and it is likely that the stock index has changed considerably over a year, which makes the forecasting challenge very hard. To generate more questions, we can think about:
Narrow ranges about mutually exclusive events, such that the underlying subject has a lot of uncertainty and complex factors (Indian economy, during globally uncertain periods like wars and mass inflation)
Narrow date ranges and numbers (stock index ranges, box office ranges)
Longer timelines (the model’s cutoff is in the past, but our knowledge cutoff is farther, so we have more information - hard questions can be generated by thinking about what has changed since the knowledge cutoff that we know but the model doesn’t).
Narrow Range Q: "In the year 2050, will the global market share of electric vehicles (battery electric and plug-in hybrid) in new car sales be between 70% and 75%, inclusive, according to the International Energy Agency?"
Complement Q: "In the year 2050, will the global market share of electric vehicles (battery electric and plug-in hybrid) in new car sales be less than 70% or greater than 75%, according to the International Energy Agency?"
Rationale: EV adoption is changing rapidly. A specific range 27 years out is inherently uncertain.
Narrow Range Q: "On December 31, 2025, will the approval rating of the US President, as measured by Gallup, be between 45% and 47%, inclusive?"
Complement Q: "On December 31, 2025, will the approval rating of the US President, as measured by Gallup, be less than 45% or greater than 47%?"
Rationale: Presidential approval ratings are volatile and influenced by many factors, and we don’t know who the President will be on December 31, 2025.
Question P (Narrow Range): "Will the average retail price of petrol in Mumbai on June 20, 2024, be between 104.50 and 104.70 Indian Rupees per liter, according to the Indian Oil Corporation?"
Question Q (Complement): "Will the average retail price of petrol in Mumbai on June 20, 2024, be less than 104.50 or greater than 104.70 Indian Rupees per liter, according to the Indian Oil Corporation?"
Rationale: Fuel prices are subject to global oil market fluctuations, government taxes, and local market conditions. Predicting a narrow price range is challenging, and these factors can change rapidly, rendering the October 2023 knowledge less useful.

AFTER completing this analysis for each example, generate questions that combine the most challenging elements identified across all examples.

                Example Analysis (using BSE Sensex question):
                Take the BSE Sensex question:
                DEEP ANALYSIS:
                Question Structure:
                Asks about a very specific range (75k-77k)
                For a specific date (July 21, 2024)
                About a complex financial instrument (Indian stock index)
                What makes this specifically challenging: a) Volatility Factor:
                The BSE Sensex is particularly volatile because:
                It's an emerging market index
                Influenced by global factors (US rates, geopolitics)
                Subject to local factors (monsoons, political changes)
                This makes narrow range predictions exponentially harder
                b) Knowledge Cutoff Impact:
                Model's knowledge ends October 2023
                India's economy has significantly changed since then:
                Interest rate cycles
                Foreign investment patterns
                Policy changes
                This creates a compound uncertainty effect
                c) Range Psychology:
                The narrow 2000-point range (75k-77k) is:
                Wide enough to seem possible
                Narrow enough to be highly improbable
                This creates a cognitive trap where the forecaster:
                Underestimates the difficulty of hitting this range
                Fails to properly calculate complementary probabilities
                d) Timeline Challenge:
                July 21, 2024 is:
                Far enough to accumulate uncertainty
                Close enough to seem predictable
                This creates false confidence in prediction ability
                To make similarly challenging questions, we need to combine:
                Highly volatile subject matter
                Narrow but plausible ranges
                Timeline that exploits knowledge cutoff
                Multiple interacting uncertainty factors
                Example application to a different domain: 
                P: "Will the Bitcoin hash rate be between 750 and 800 exahashes/second on December 31, 2027, according to BitInfoCharts?"
                Q: "Given that Bitcoin  has reached $150k by 2028, Will Ethereum's largest one day move exceed 25% by 2028"?
                This captures the same challenging elements:
                Volatile subject (Bitcoin mining)
                Narrow but plausible range
                Multiple uncertainty factors (energy prices, mining difficulty, regulations, multiple cryptocurrencies)
                Misunderstanding of Conditional Probabilities
                Timeline beyond knowledge cutoff
"""

llama_in_context_examples = """
Example 1:
Question P:Title: Will Joe Biden die before Saturday?Body: Resolves YES if there is an official announcement of Joe Biden's death before market close at the end of Friday, July 26th.
Otherwise, resolves NO.Forecast: 0.0005Reasoning:Reasoning: To estimate the probability of Joe Biden dying before Saturday, I will consider his age, health, and available information. As of my knowledge cutoff in 2023, Joe Biden was 80 years old. While his age is a risk factor for mortality, I do not have any information about his current health status or any life-threatening conditions. Given the lack of specific information about his health, I will rely on general mortality rates for people of his age group. According to the Social Security Administration's Actuarial Life Table, the probability of a person dying within a year at age 80 is around 7.5%. However, this is a yearly probability, and we are looking at a much shorter timeframe (a few days). To adjust for this, I will use a rough estimate of the daily mortality rate, which is approximately 1/365 of the yearly rate. This would put the daily mortality rate at around 0.0205% (7.5%/365). Since we are looking at a 2-day period (Thursday and Friday), I will multiply this rate by 2, resulting in a probability of approximately 0.041% or 0.00041. However, considering the nature of the question and the need to provide a more calibrated estimate, I will adjust this probability slightly upward to account for any unforeseen circumstances.
Probability: 0.0005
Title: Will Kamala Harris assume the role of President within 2 weeks of Biden's death?Body: This question resolves as YES if Kamala Harris is officially inaugurated as the President of the United States within 14 days following an official announcement of Joe Biden's death. Otherwise, it resolves as NO. This question focuses on the swift constitutional transition of power and the readiness of the vice presidency to assume command, which is crucial for maintaining governmental stability and projecting confidence both domestically and internationally.Forecast: 0.95Reasoning:Reasoning: The 25th Amendment to the United States Constitution outlines the order of presidential succession, with the Vice President being the first in line to assume the presidency in the event of the President's death or incapacitation. Given this constitutional framework, Kamala Harris, as the current Vice President, would be next in line to assume the presidency.
Question R:Title: Will either Joe Biden die before Saturday or Kamala Harris assume the role of President within 2 weeks of Biden's death?Body: This question resolves as YES if either of the following occurs:
(a) There is an official announcement of Joe Biden's death before market close at the end of Friday, July 26th.
(b) Kamala Harris is officially inaugurated as the President of the United States within 14 days following an official announcement of Joe Biden's death.
If neither of these events occurs, the question resolves as NO. The resolution criteria for each component are as follows:
* For component (a), the resolution is based on the timing of the official announcement of Joe Biden's death.
* For component (b), the resolution is based on the official inauguration of Kamala Harris as President within the specified timeframe following Biden's death.Forecast: 0.02Reasoning:Reasoning: To estimate the probability of this question resolving as YES, we need to consider the likelihood of two separate events: (a) Joe Biden's death before Saturday, July 26th, and (b) Kamala Harris assuming the role of President within 2 weeks of Biden's death.
First, let's consider event (a). As of my knowledge cutoff in 2023, Joe Biden was 80 years old. While his age puts him at a higher risk of mortality, there is no publicly available information suggesting any severe health issues that would significantly increase the likelihood of his death within a specific timeframe. Given the lack of information about his current health status, it's challenging to estimate the probability of his death before July 26th. However, considering his age and the general mortality rate, it's reasonable to assume that the probability of his death within a short period is relatively low.
Now, let's consider event (b). If Joe Biden were to pass away, Kamala Harris, as the Vice President, would be next in line to assume the presidency. The presidential transition process in the United States is well-established, and it's likely that Harris would be inaugurated as President within a short period following Biden's death. The 25th Amendment to the US Constitution outlines the order of presidential succession, and the process of presidential transition is typically swift.
Given these considerations, the probability of event (b) occurring, assuming event (a) happens, is relatively high. However, since event (a) is the primary condition for event (b) to occur, the overall probability of the question resolving as YES is still relatively low.
Probability: 0.02

Expert Analysis of Flaws:	
1	DEEP STRUCTURAL ANALYSIS
* Question Structure:
    * P = Biden's death by Saturday (extremely rare event)
    * Q = Harris assumes presidency within 2 weeks of death (conditional event)
    * R = P OR Q (compound event with dependency)
    * Clear resolution criteria with specific timeframes
    * Involves conditional probability with dependent events
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Conditional Probability Confusion
    * Quote: "However, since event (a) is the primary condition for event (b) to occur, the overall probability of the question resolving as YES is still relatively low."
    * Yet assigns P(P OR Q) = 0.02 > P(P) = 0.0005
    * Mathematical impossibility: Since Q can only occur if P occurs, P(P OR Q) must equal P(P)
    * Fails to recognize that Q is entirely dependent on P
* Secondary Flaw: Base Rate Adjustment
    * Incorrectly adjusts base probability upward when combining events
    * Quote: "Given these considerations, the probability of event (b) occurring, assuming event (a) happens, is relatively high"
    * Uses high conditional probability of Q|P to inflate P(P OR Q)
3. UNCERTAINTY MULTIPLICATION FACTORS
* Multiple probabilistic confusions:
    * Confusion between conditional and joint probability
    * Misunderstanding of dependent events
    * Base rate neglect when combining probabilities
    * Failure to recognize logical dependencies
4. PATTERN REPLICATION GUIDE To replicate this forecasting flaw:
* Create questions involving:
    * An extremely rare primary event
    * A highly likely conditional event
    * Logical dependency between events
    * OR statements combining dependent events
    * Time-sensitive resolutions
5. TRIGGERS
* Key triggers that expose these flaws:
    * Compound events with dependencies
    * Very low probability primary events
    * High probability conditional events
    * OR statements that mask dependencies
    * Time-sensitive resolution criteria
6. EXAMPLE GENERATION New question that would trigger similar flaws:
P: "Will Apple achieve a $5 trillion market cap by October 2025?” Q: "Will Donald Trump’s collective funding for his campaign exceed $2.?” Q: "Will the SPX index exceed 7500 points by December 2025?“:
* Rare primary event (AAPL $5T market cap)
* Conditional follow-up event (stock price movement), but this is subtle because AAPL $5T doesn’t directly imply SPX index
* Logical dependency (SPX index likely to soar due to tech stock correlations)
* Specific timeframes such that one event is just before the other 
The most severe error in this forecaster's reasoning is failing to recognize that when Q can only occur if P occurs, P(P OR Q) must equal P(P). This leads to systematic overestimation of compound probabilities involving rare primary events with likely conditional outcomes.

Example 2:
Question P:Title: On July 17, 2024, will Nvidia's market capitalization be larger than Apple's?Body: This question resolves Yes if, on July 17, 2024, the market cap of Nvidia is greater than that of Apple, according to Companies Market Cap, when checked by Metaculus at approximately 4 PM EDT (8 PM UTC).Forecast: 0.05Reasoning:Reasoning: To estimate the probability of Nvidia's market capitalization being larger than Apple's on July 17, 2024, we need to consider several factors. First, we should look at the current market capitalization of both companies. As of my knowledge cutoff in 2023, Apple's market capitalization is significantly larger than Nvidia's. However, Nvidia has been growing rapidly in recent years, driven by the increasing demand for graphics processing units (GPUs) and artificial intelligence (AI) technologies.
Another factor to consider is the growth prospects of both companies. Nvidia is well-positioned to benefit from the growing demand for AI, autonomous vehicles, and cloud computing, which could drive its revenue and market capitalization higher. On the other hand, Apple's growth prospects are more uncertain, as the company faces intense competition in the smartphone market and is heavily dependent on a few products.
Despite Nvidia's growth prospects, it's still a long shot for the company's market capitalization to surpass Apple's in just over a year. Apple's market capitalization is currently over $2 trillion, while Nvidia's is around $500 billion. To surpass Apple's market capitalization, Nvidia would need to experience extremely rapid growth, which is unlikely.
Additionally, the stock market is inherently unpredictable, and many factors can affect a company's market capitalization, including economic conditions, regulatory changes, and unexpected events.
Considering these factors, I estimate that the probability of Nvidia's market capitalization being larger than Apple's on July 17, 2024, is relatively low.
Probability: 0.05
Question Q:Title: On July 17, 2024, will Nvidia's market capitalization be less than or equal to Apple's?Body: This question resolves Yes if, on July 17, 2024, the market cap of Nvidia is less than or equal to that of Apple, according to Companies Market Cap, when checked by Metaculus at approximately 4 PM EDT (8 PM UTC).Forecast: 0.12Reasoning:Reasoning: To estimate the probability of Nvidia's market capitalization being less than or equal to Apple's on July 17, 2024, we need to consider the current market trends and the historical performance of both companies. As of my knowledge cutoff in 2023, Nvidia's market capitalization was around $540 billion, while Apple's was around $2.3 trillion. Given the significant difference in their market capitalizations, it's essential to assess the factors that could lead to a change in this gap.
Several factors could influence the market capitalization of both companies, including their financial performance, product releases, competition, and overall market conditions. Nvidia has been growing rapidly due to the increasing demand for graphics processing units (GPUs) in gaming, artificial intelligence, and data centers. However, Apple's market capitalization is more than four times that of Nvidia, making it challenging for Nvidia to catch up in the short term.
That being said, the technology sector is known for its volatility, and significant changes can occur rapidly. If Nvidia continues to innovate and expand its market share, while Apple faces challenges or experiences a decline in its stock price, it's possible that the gap between their market capitalizations could narrow.
Considering these factors, I estimate that the probability of Nvidia's market capitalization being less than or equal to Apple's on July 17, 2024, is relatively low, but not impossible.
Probability: 0.12

Expert Analysis about flaws:

1. DEEP STRUCTURAL ANALYSIS
- Question Structure:
  * P: NVDA market cap > AAPL market cap (rare event)
  * Q: NVDA market cap ≤ AAPL market cap (complement event)
  * Specific date and time (July 17, 2024, 4 PM EDT)
  * Involves comparison of two numerical quantities
  * Knowledge cutoff impact on fast-moving tech sector
* NVDA market cap being less than AAPL seems so obvious that the forecaster mixes up the reasoning in Question Q, argues for NVDA most likely being less than AAPL, yet provides the forecast = 0.12 by mixing up the probability, because the event seems too obvious.

2. REASONING FLAW IDENTIFICATION
- Primary Flaw: Probability Consistency Violation
  * P(NVDA > AAPL) = 0.05
  * P(NVDA ≤ AAPL) = 0.12
  * Sum = 0.17, violating P(A) + P(not A) = 1
  * Quote: "relatively low, but not impossible" shows qualitative reasoning trumping mathematical necessity

- Secondary Flaw: Anchoring to Outdated Data, Cutoff date can be exploited
  * Quote: "As of my knowledge cutoff in 2023, Nvidia's market capitalization was around $540 billion"
  * Fails to acknowledge uncertainty about current values
  * Uses outdated ratios ("more than four times") to make future predictions

3. UNCERTAINTY MULTIPLICATION FACTORS
- Temporal Uncertainty:
  * Knowledge cutoff (2023) vs prediction date (2024)
  * AI industry dynamics changing rapidly
  * Market volatility in tech sector
- Numerical Uncertainty:
  * Uncertain about current market caps
  * Complex growth rate calculations
  * Multiple interacting market factors

4. PATTERN REPLICATION GUIDE
To replicate this forecasting flaw:
- Create questions involving:
  * Fast-moving industries with recent major changes
  * Numerical comparisons of two quantities
  * Incumbent leader vs rapid growth challenger
  * Time periods spanning beyond knowledge cutoff
  * Sectors with exponential growth potential
- Comparison between entities such that one is obviously much better than the other, so that the forecaster gets confused and mixes up negation probability

5. TRIGGERS
- Key triggers that expose these flaws:
  * Complementary events with numerical thresholds but without mentioning the numbers (NVDA vs AAPL market cap but not mentioning the market cap)
  * Rapidly evolving market dynamics
  * Knowledge cutoff exploitation
  * Comparison of established vs emerging players
  * Quantitative predictions requiring current data
- Comparison of entities such that one is obviously far ahead 

6. EXAMPLE GENERATION
Questions that would trigger similar flaws:

A. Emerging vs Established Markets
P: "Will India's total semiconductor production exceed Taiwan's by December 2025?"
Q: "Will India's total semiconductor production be less than or equal to Taiwan's by December 2025?"
* Exploits knowledge cutoff (recent semiconductor policy changes)
* Compares established leader vs emerging player
* Requires numerical comparison (but not explicitly mentioning the numbers)
* Fast-moving industry dynamics

B. Currency Market Disruption
P: "Will the total market cap of stablecoins exceed Japan's M1 money supply by June 2025?"
Q: "Will the total market cap of stablecoins be less than or equal to Japan's M1 money supply by June 2025?"
* Compares traditional vs emerging financial systems
* Involves rapidly changing market dynamics
* Requires current numerical data (but not explicitly mentioning the numbers)
* Complex interaction of multiple factors

C. Complex numerical quantities without actually mentioning them
P:  Will the quantity (Solana price in USD x ETH market cap in USD) exceed BTC market cap by December 2026?
Q: Will the quantity (Solana price in USD x ETH market cap in USD) be less than the BTC market cap by December 2026? 

The forecaster's most critical error is the probability inconsistency, compounded by over-reliance on outdated data in a rapidly evolving sector, and trying to estimate multiple quantities for comparison when none are provided. This suggests a systematic bias toward qualitative reasoning over mathematical consistency, particularly when dealing with fast-moving markets beyond the knowledge cutoff date, and mixing up entities while comparing them.Example 3:

Consistency Type: expevidence

Consistency Definition: Pr(P) = Pr(P|Q)Pr(Q) + Pr(P|¬Q)(1 - Pr(Q)), Explanation: Tuple is (P, Q, P|Q, P|¬Q)

Question P:
Title: Will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th?
Body: Politico reports:

Two safe-seat members have already publicly urged Biden out. Two of the most endangered Democrats told local news outlets they’ve already written off Biden’s chances to win in November.

Behind the scenes, things are even more frenzied. There are multiple drafts of letters circulating among House Democrats and at least one would call on Biden to end his campaign, according to five people familiar with the efforts.

This market resolves YES if a group of at least five named Democratic Party US House Members/US Senators work together to ask Joe Biden to consider ending his 2024 re-election campaign, or to urge him to do so, before Saturday July 20th (Pacific Time).
Forecast: 0.72
Reasoning:
Reasoning: The Politico report provides strong evidence that there is significant discontent among Democratic Party members regarding Biden's chances of winning in November. Two safe-seat members have already publicly urged Biden to drop out, and multiple drafts of letters are circulating among House Democrats, with at least one calling for Biden to end his campaign. Additionally, two endangered Democrats have written off Biden's chances, indicating a growing sense of desperation among party members. Given the urgency and the fact that multiple drafts of letters are already circulating, it is likely that a group of at least five named Democratic Party US House Members/US Senators will work together to ask Biden to consider ending his campaign before July 20th. The fact that this market requires a group effort, rather than individual statements, increases the likelihood of a coordinated effort to pressure Biden to drop out.

Probability: 0.72

Question Q:
Title: Will a new Democratic contender officially enter the 2024 presidential race by July 20th?
Body: 
Forecast: 0.42
Reasoning:
Reasoning: Considering the current state of the 2024 presidential election, several factors contribute to the likelihood of a new Democratic contender entering the race by July 20th. Firstly, the Democratic primary field is relatively small compared to previous elections, which might create an opportunity for a new candidate to enter and gain traction. Additionally, the primary debates and caucuses are still months away, leaving room for a late entrant to make an impact. However, it's also worth noting that the Democratic National Committee (DNC) has already set rules and deadlines for candidates to qualify for the debates, which might discourage late entries. Furthermore, the existing candidates have already established themselves and built momentum, making it more challenging for a new contender to gain significant support. Given these factors, I believe there is a moderate chance of a new Democratic contender entering the race by July 20th.

Probability: 0.42

Question R:
Title: Given that a new Democratic contender officially enters the 2024 presidential race by July 20th, will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th?
Forecast: 0.18
Reasoning:
Reasoning: Given the scenario of a new Democratic contender entering the 2024 presidential race by July 20th, the likelihood of a group of at least 5 named Democratic House Members or US Senators pressuring Biden to drop out before July 20th seems relatively low. This is because Joe Biden has already announced his intention to run for re-election, and the Democratic Party has generally shown support for his candidacy. Additionally, the party's leadership and many prominent members have expressed their backing for Biden's re-election bid.

However, the entry of a new contender could potentially create some divisions within the party, especially if the new candidate gains significant traction or has a compelling message that resonates with some party members. 

Considering these factors, I would estimate the probability of this event occurring as relatively low, but not impossible, given the uncertainties of politics and the potential for unexpected developments.

Probability: 0.18

Question S:
Title: Given that no new Democratic contender officially enters the 2024 presidential race by July 20th, will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th?
Forecast: 0.12
Reasoning:
Reasoning: Given the condition that no new Democratic contender officially enters the 2024 presidential race by July 20th, the likelihood of a group of at least 5 named Democratic House Members or US Senators pressuring Biden to drop out before July 20th seems low. This is because, historically, party leaders and prominent members tend to rally behind the incumbent president, especially when there is no strong alternative candidate. Additionally, the lack of a new challenger by July 20th may indicate that the party is generally satisfied with Biden's candidacy or that potential challengers are hesitant to enter the race. However, there could be internal party dynamics or concerns about Biden's electability that might lead some members to pressure him to drop out. Considering these factors, I would estimate the probability of this event occurring as relatively low.

Probability: 0.12

1. DEEP STRUCTURAL ANALYSIS
* Question Structure: P: "Will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th?" [Context framed with: "Two safe-seat members have already publicly urged Biden out... Two endangered Democrats... multiple drafts of letters circulating"] Q: "Will a new Democratic contender officially enter the 2024 presidential race by July 20th?"
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Narrative-Driven Probability Inflation (0.72)
    * Uses "Two safe-seat members" to make five members seem achievable
    * Exploits "multiple drafts" to suggest imminent action
    * Leverages both public (2 members) and private signals ("behind the scenes... frenzied")
    * Creates false sense of momentum by combining:
        * Existing public dissent (2 members)
        * Private discussions (multiple drafts)
        * Endangered Democrats' concerns
        * "Behind the scenes" urgency
* Secondary Flaw: Correlation Probability Inversion
    * P(P) = 0.72 (unconditional)
    * P(P|Q) = 0.18 (given new contender)
    * P(P|¬Q) = 0.12 (given no new contender)
    * Violates law of total probability: 0.72 ≠ 0.18(0.42) + 0.12(0.58)
    * Illogically assigns lower probability when positive catalyst (new contender) occurs
3. UNCERTAINTY MULTIPLICATION FACTORS
* Multiple Requirements:
    * Named members (not anonymous)
    * Group coordination (single message)
    * Public disclosure
    * Specific timeline
    * Political risk factors
4. PATTERN REPLICATION GUIDE To trigger similar errors:
* Frame with existing partial progress
* Combine public and private signals
* Suggest mounting momentum, although the event is highly unlikely
* Create urgency narrative
* Use specific recent examples
5. TRIGGERS
* Key elements that expose these flaws:
    * Progress framing ("already" language)
    * Multiple evidence types (public + private)
    * Group coordination requirements
    * Time pressure elements
    * Correlation with catalyzing events
6. EXAMPLE GENERATION A better example following this pattern:
P: "Will SEC approve 5+ new spot crypto ETFs by July 2024? [Note: SEC has already approved Bitcoin ETFs, and sources report multiple applications in final review stages. Two commissioners have publicly supported broader crypto ETF approval.]" Q: "Will crypto market cap exceed $3T by May 2024?"
* Uses existing approval to make additional ones seem likely, although the event is highly unlikely
* Combines public signals (approvals) with private ones (applications in review)
* Creates false sense of momentum
* Requires multiple coordinated actions
* Natural correlation between events
The forecaster's critical error is allowing narrative framing and recent examples to override proper probability assessment, particularly in:
1. Using partial progress to inflate probability of full completion
2. Failing to properly weight difficulty of coordinated action
3. Mishandling correlation between related events
4. Letting urgency narrative override mathematical consistency
NOTE: the momentum/seemingly false context should be grounded in truth or facts and should not be made up. It should only be provided as misleading context, only for the purpose of training the forecaster so that they don’t make the same mistake again, but ground it in facts.

Example 4:
Consistency Definition: Pr(P∨Q) = Pr(P) + Pr(¬P∧Q), Explanation: Tuple is (P, ¬P∧Q, P∨Q)
Question P:Title: Will the maximum weekly rate of hospitalizations per 100,000 in the US occur within four weeks of the combined peak for each of COVID, influenza, and RSV in the 2023-24 season?.

Probability: 0.8
Question Q:Title: Will both the maximum weekly rate of hospitalizations per 100,000 in the US NOT occur within four weeks of the combined peak for each of COVID, influenza, and RSV in the 2023-24 season AND the influenza vaccination rate for adults in the US exceed 50% by the end of the 2023-24 season?
Reasoning: To estimate the probability of both conditions being met, we need to analyze each condition separately and then combine them.
For condition (a), the maximum weekly rate of hospitalizations per 100,000 for any of COVID, influenza, or RSV not occurring within four weeks of the combined peak for each of these diseases in the 2023-24 season, we can look at historical data. Typically, the peak for influenza and RSV occurs between December and February, while COVID-19 has had varying peaks. However, given the overlap of these diseases, it's possible that the maximum weekly rate of hospitalizations for one of them might not occur exactly at the same time as the combined peak. Considering the variability in the timing of peaks and the fact that the question allows for a four-week window, I would estimate the probability of this condition being met as relatively high.
For condition (b), the influenza vaccination rate for adults in the US exceeding 50% by the end of the 2023-24 season, we can look at historical vaccination rates. According to the CDC, the influenza vaccination rate for adults in the US has been steadily increasing over the years, with the 2020-21 season reaching around 48.4% and the 2021-22 season reaching around 49.2%. Given this trend, it's possible that the vaccination rate could exceed 50% in the 2023-24 season, especially with increased awareness and efforts to promote vaccination.
Combining these two conditions, we need to consider the correlation between them. While there might be some correlation between the severity of the influenza season (which could affect vaccination rates) and the timing of peaks, it's not a direct causal relationship. Therefore, we can assume a relatively low correlation between the two conditions.
Given these considerations, I would estimate the probability of both conditions being met as follows:
Probability: 0.62
Question R:Title: Will either the maximum weekly rate of hospitalizations per 100,000 in the US occur within four weeks of the combined peak for each of COVID, influenza, and RSV, or will the influenza vaccination rate for adults in the US exceed 50% by the end of the 2023-24 season?
Probability: 0.72

Expert Analysis:
1. DEEP STRUCTURAL ANALYSIS
* Question Components:
    * P: Hospital peaks within 4 weeks (0.8)
    * ¬P∧Q: Not within 4 weeks AND 50% vaccination (0.62)
    * P∨Q: Either peaks within 4 weeks OR 50% vaccination (0.72)
    * Complex resolution criteria with epidemiological weeks
    * Multiple health metrics interaction
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Logical Operator Inconsistency
    * P∨Q = P + (¬P∧Q) violated
    * P = 0.8, ¬P∧Q = 0.62, P∨Q = 0.72
    * Mathematical impossibility: 0.72 < 0.8 + 0.62
    * Quote: "natural overlap of peak hospitalization rates" used to justify probabilities without considering logical constraints
* Secondary Flaw: Conjunction Fallacy
3. UNCERTAINTY MULTIPLICATION FACTORS
* Multiple Data Complexities:
    * Three different viruses
    * Multiple peak times
    * Vaccination rate tracking
    * Reporting delays
    * Data revision periods
4. PATTERN REPLICATION GUIDE To trigger similar errors:
* Create questions involving:
    * Multiple time-series peaks
    * Percentage thresholds
    * Complex measurement criteria
    * Data reporting conditions
    * Logical operators with health metrics
5. TRIGGERS
* Key elements that expose these flaws:
    * Multiple peak alignment requirements
    * Percentage-based thresholds
    * Combined logical operators
    * Complex resolution criteria
    * Public health data dependencies
6. EXAMPLE GENERATION Question set that would trigger similar flaws:
P: "Will US EV sales peak within 4 weeks of combined peak charging station installations across top 5 states in 2024?”, Q: “Will Tesla's market share exceed 50% and SPX hit an all time high in 2024?”
* Multiple time series peaks
* Percentage threshold
* Complex measurement criteria
* Data reporting dependencies
* Logical operator confusion

Example 5:
Question P:Title: Will Russian athletes be barred from competing at the 2024 Olympics?
Reasoning:Reasoning: The International Olympic Committee (IOC) has been under pressure to take a stance on Russian participation in the Olympics due to the ongoing conflict in Ukraine. In 2022, the IOC recommended that international sports federations and event organizers not invite or allow the participation of Russian and Belarusian athletes and officials. However, the IOC also stated that it would be open to Russian athletes competing under a neutral flag.
In the 2020 Olympics, Russian athletes competed under the Russian Olympic Committee (ROC) flag due to a doping ban. However, the current situation is different, and the IOC might take a stronger stance against Russian participation. The IOC has been discussing the possibility of Russian athletes competing under a neutral flag, but it's unclear if this will be allowed.
Given the current situation, it's likely that Russian athletes will face some restrictions on their participation in the 2024 Olympics. However, it's also possible that some Russian athletes might be allowed to compete under a neutral flag, which could lead to more than 34 Russian athletes participating.
Considering these factors, I estimate that the probability of Russian athletes being barred from competing at the 2024 Olympics is relatively high, but not certain.
Probability: 0.7
Question Q:Title: Will more than 34 Russian athletes compete at the 2024 Olympics?Body: This question will resolve as Yes if more than 34 Russian athletes compete at the 2024 Olympics under any flag, including (i) the Russian flag, (ii) The Russian Olympic Committee's flag, or (iii) any other independent or neutral flag. This will resolve based on the official list of countries and athletes present at the 2024 Olympics or based on credible media reports
Reasoning: Additionally, the International Olympic Committee (IOC) has shown a willingness to allow Russian athletes to compete under various flags, including the ROC flag and neutral flags. Considering these factors, it is reasonable to assume that more than 34 Russian athletes will compete at the 2024 Olympics.
Probability: 0.85
1. DEEP STRUCTURAL ANALYSIS
* Question Structure:
    * P: Russians barred (≤34 athletes) (0.7)
    * ¬P: Russians not barred (>34 athletes) (0.85)
    * Clear violation of P + ¬P = 1
    * High-stakes geopolitical context
    * Historical vs current dynamics tension
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Temporal Inconsistency
    * Historical Anchor: "At the 2020 Olympics, 335 Russian athletes competed"
    * Current Reality: "ongoing conflict in Ukraine"
    * Fails to properly weight unprecedented geopolitical situation
    * Quote: "WADA ban has been partially lifted" shows fixation on past regulatory framework
* Secondary Flaw: Bias-Driven Double Counting
    * Sees strong case for both outcomes:
        * Pro-participation: Historical precedent, IOC flexibility
        * Anti-participation: War in Ukraine, international pressure
    * Results in P + ¬P = 1.55 (0.7 + 0.85)
    * Reflects cognitive dissonance in handling controversial topic
3. UNCERTAINTY MULTIPLICATION FACTORS
* Geopolitical Complexity:
    * War dynamics
    * International sanctions
    * Sports diplomacy
    * IOC politics
    * National pride factors
4. PATTERN REPLICATION GUIDE To trigger similar biases:
* Questions involving:
    * Nations with complex international relations
    * Historical precedent vs current crisis
    * Numerical thresholds with political significance
    * International organization decision-making
    * High-stakes diplomatic outcomes
5. TRIGGERS
* Key elements exposing these flaws:
    * Historical data contradicting current reality
    * Politically charged decisions
    * Multiple stakeholder influences
    * Binary outcomes with continuous underlying variable
    * International relations complexity
6. EXAMPLE GENERATION Question pair that would trigger similar biases:
P: "Will Chinese companies constitute less than 10% of new US semiconductor fabs in 2025?" ¬P: "Will Chinese companies constitute 10% or more of new US semiconductor fabs in 2025?"
* Historical data (previous Chinese investment levels)
* Current geopolitical tensions (tech restrictions)
* Multiple stakeholder influences (CFIUS, state governments)
* Numerical threshold with political significance
* International relations complexity
The forecaster's critical errors stem from:
1. Inability to reconcile historical precedent with unprecedented current events
2. Cognitive dissonance in handling politically charged topics
3. Bias toward seeing strong arguments for both outcomes
4. Over-reliance on past regulatory frameworks for current political decisions
This case particularly highlights how forecasters can struggle when:
* Historical data points toward one outcome
* Current geopolitical reality suggests another
* Topic involves potential biases about international relations
* Decision involves multiple competing political pressures
The key insight is how the forecaster's attempt to acknowledge both historical precedent and current reality led to probability assignments that violate basic laws, reflecting deeper uncertainty about how to weight unprecedented geopolitical factors. So, generate questions such that the event is likely due to historical precedent, is about a controversial/bias-triggering topic, and has high-stakes, rapidly changing dynamics recently that make the event unlikely.
"""


deepseek_in_context_examples = """ Example 1:
Question P:Title: Will Borussia Dortmund win the 2023/2024 UEFA Champions League?Body: The question resolves Yes if Borussia Dortmund wins the Final on the 1st of June 2024, according to UEFA. All other teams will resolve as No.Forecast: 0.08Reasoning:Reasoning: Borussia Dortmund is a strong team with a history of competitive performances in the UEFA Champions League. However, winning the tournament requires not only consistent high-level play but also favorable draws, minimal injuries, and some degree of luck. The competition includes other elite teams like Manchester City, Bayern Munich, Real Madrid, and Paris Saint-Germain, all of whom are also strong contenders. While Dortmund has the potential to reach the later stages of the tournament, the probability of them winning the entire competition is relatively low given the depth and quality of the field. Based on their recent performances, squad strength, and the competitive landscape, I estimate their chances of winning the 2023/2024 UEFA Champions League as modest but not negligible. Probability: 0.08
Question Q:Title: Will Borussia Dortmund fail to win the 2023/2024 UEFA Champions League AND will Jude Bellingham be sidelined for more than 4 UEFA Champions League matches due to injury in the 2023/2024 season? Body: This question resolves as Yes if both of the following conditions are met: (a) Borussia Dortmund does not win the Final of the 2023/2024 UEFA Champions League on the 1st of June 2024, according to UEFA. If any other team wins, or if the competition is not completed, this part resolves as Yes. (b) Jude Bellingham, a key player for Borussia Dortmund, misses more than four UEFA Champions League matches in the 2023/2024 season due to an injury, as officially reported by Borussia Dortmund or media outlets focusing on the UEFA Champions League. If he misses four or fewer matches due to injury or does not miss any matches for this reason during the season, this part resolves as No. The combined question resolves as No if either or both of these conditions are not met.Body: This question resolves as Yes if both of the following conditions are met: (a) Borussia Dortmund does not win the Final of the 2023/2024 UEFA Champions League on the 1st of June 2024, according to UEFA. If any other team wins, or if the competition is not completed, this part resolves as Yes. (b) Jude Bellingham, a key player for Borussia Dortmund, misses more than four UEFA Champions League matches in the 2023/2024 season due to an injury, as officially reported by Borussia Dortmund or media outlets focusing on the UEFA Champions League. If he misses four or fewer matches due to injury or does not miss any matches for this reason during the season, this part resolves as No. The combined question resolves as No if either or both of these conditions are not met.Forecast: 0.85Reasoning:Reasoning: To assess the probability of this question resolving as Yes, we need to evaluate the likelihood of both conditions being met independently and then combine them.
1. Borussia Dortmund failing to win the 2023/2024 UEFA Champions League: The UEFA Champions League is a highly competitive tournament with many strong teams, including Manchester City, Bayern Munich, Real Madrid, and others. Borussia Dortmund, while a strong team, is not historically the most dominant in this competition. The probability of them not winning is high, likely around 0.85.
2. Jude Bellingham being sidelined for more than 4 UEFA Champions League matches due to injury: Jude Bellingham is a key player for Dortmund, and his injury history is relatively limited. However, injuries are unpredictable, and the physical demands of the Champions League increase the risk.  the likelihood of this happening is around 0.25.
To combine these probabilities, we multiply them since both conditions must be met for the question to resolve as Yes:0.85 (Dortmund not winning) × 0.25 (Bellingham missing >4 matches) = 0.2125.
Probability: 0.2125
Question R:Title: Will either Borussia Dortmund win the 2023/2024 UEFA Champions League or Jude Bellingham be sidelined for more than 4 UEFA Champions League matches due to injury in the 2023/2024 season?Body: This question resolves as Yes if either of the following occurs:(a) Borussia Dortmund wins the Final of the 2023/2024 UEFA Champions League on the 1st of June 2024, according to UEFA.(b) Jude Bellingham, a key player for Borussia Dortmund, misses more than four UEFA Champions League matches in the 2023/2024 season due to an injury, as officially reported by Borussia Dortmund or media outlets focusing on the UEFA Champions League.If neither of these events occurs, the question resolves as No.Forecast: 0.1Reasoning:Reasoning: To assess the probability of this question resolving YES, we need to evaluate the likelihood of either Borussia Dortmund winning the 2023/2024 UEFA Champions League or Jude Bellingham being sidelined for more than 4 matches due to injury.
1. Borussia Dortmund winning the UEFA Champions League: The UEFA Champions League is highly competitive, with several elite teams (e.g., Manchester City, Bayern Munich, Real Madrid) consistently performing at a high level.  Based on recent trends and team strength, the probability of Dortmund winning the tournament is relatively low, perhaps around 0.10 (10%).
2. Jude Bellingham being sidelined for more than 4 matches due to injury: Bellingham is a key player for Dortmund, and his injury history and playing style suggest some risk of injury. However, missing more than 4 matches in a single Champions League season is a significant absence. Assuming a typical injury risk for a high-profile player, the probability of this occurring might be around 0.15 (15%).
To calculate the combined probability of either event occurring, we use the formula for the probability of either A or B happening:P(A or B) = P(A) + P(B) - P(A) * P(B)Plugging in the values:P(A or B) = 0.10 + 0.15 - (0.10 * 0.15) = 0.25 - 0.015 = 0.235
Thus, the probability of the question resolving YES is approximately 0.235 (23.5%).
Probability: 0.235

Expert Analysis:

1. DEEP STRUCTURAL ANALYSIS
* Question Components:
    * P: BVB wins Champions League (0.08, later implied 0.10)
    * ¬P∧Q: BVB doesn't win AND Bellingham injured >4 matches (0.85)
    * P∨Q: BVB wins OR Bellingham injured (0.235)
    * Multiple competing elite teams
    * Individual player injury risk
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Base Rate Inconsistency
    * Different P values in different contexts:
        * P = 0.08 when asked directly
        * P = 0.10 when calculating OR probability
        * Implied P = 0.15 in joint probability calculations
    * Quote: "while a strong team, is not historically the most dominant" used inconsistently
* Secondary Flaw: Individual vs Team Event Confusion
    * Bellingham injury probability varies dramatically:
        * 0.25 in one calculation
        * 0.15 in another
    * Fails to maintain consistent view of individual player risk
    * Quote: "injury history is relatively limited" vs "typical injury risk" shows inconsistent baseline
3. UNCERTAINTY MULTIPLICATION FACTORS
* Multiple Complexity Layers:
    * Team competition dynamics
    * Individual player health
    * Tournament format effects
    * Historical precedents
    * Squad depth considerations
4. PATTERN REPLICATION GUIDE To trigger similar errors:
* Create questions involving:
    * Multiple elite competitors
    * Key player dependencies
    * Historical vs current performance
    * Individual vs team outcomes
    * Tournament format complexities
5. TRIGGERS
* Key elements exposing these flaws:
    * Elite competition with multiple favorites
    * Star player injury risk
    * Tournament knockout format
    * Historical performance data
    * Player importance weighting
6. EXAMPLE GENERATION Questions that would trigger similar biases:
P: "Will the Golden State Warriors win the 2024-25 NBA Championship? [Note: Warriors have won 4 recent titles but face strong competition from Nuggets, Celtics, and multiple contenders]" Q: "Will Warriors fail to win AND will Steph Curry miss >20 games due to injury?" 
P: “Will India win the next T20 Cricket World Cup?”
Q: “Will India fail to win the next T20 Cricket World Cup AND will K L Rahul miss >3 games due to injury?
* Multiple elite competitors (like Champions League or Cricket World Cup)
* Star player dependency (like Bellingham or K L Rahul)
* Historical success vs current competition
* Individual vs team outcome interaction
* Tournament format considerations
The forecaster's critical errors stem from:
1. Inability to maintain consistent probability estimates when considering the same event in different contexts
2. Confusion between individual and team event probabilities
3. Overcomplication of simple logical relationships
4. Inconsistent weighting of historical vs current factors
This case particularly highlights how forecasters can struggle when:
* Multiple elite competitors make base rates unclear
* Individual player outcomes affect team probabilities
* Historical data provides conflicting signals
* Multiple probability calculations required across questions

Example 2:
Question P:Title: Will INC form the government after the next Indian general election in 2024?Body: This question will resolve as Yes if the INC has a member sworn in as the next Indian Prime Minister following the next general elections. All other parties will resolve as No.
This question will resolve for the next general election occurring after June 1, 2022 (currently expected for May 2024).Forecast: 0.25Reasoning:Reasoning: The Indian National Congress (INC) faces significant challenges in forming the government after the 2024 general elections. The Bharatiya Janata Party (BJP), led by Prime Minister Narendra Modi, has maintained a strong electoral presence since 2014, securing a majority in both the 2014 and 2019 elections. 
Furthermore, the INC's leadership and organizational structure have been criticized for being less effective compared to the BJP's well-oiled machinery. While the INC has made efforts to rejuvenate its image and strategy, it remains to be seen whether these efforts will translate into significant electoral gains.
Given these factors, the likelihood of the INC forming the government in 2024 appears low, though not impossible. The probability is further influenced by potential shifts in public sentiment, economic conditions, and unforeseen political developments closer to the election.
Probability: 0.25
Question Q:Title: Will both the INC fail to form the government after the next Indian general election in 2024 AND India's unemployment rate be below 5% in the first quarter of 2024?Body: This question will resolve as Yes if both of the following conditions are met: (a) The INC does not have a member sworn in as the next Indian Prime Minister following the next general elections, which are expected to occur in May 2024. If any other party has a member sworn in as Prime Minister, this condition is satisfied. (b) India’s official unemployment rate, as reported by the Ministry of Labour and Employment or any other recognized government body, is below 5% for any month in the first quarter of 2024. If both conditions are met, the question resolves as Yes; otherwise, it resolves as No.Forecast: 0.85Reasoning:Reasoning: To assess the probability of this question resolving YES, we need to evaluate both conditions independently and then consider their joint likelihood.
1. INC failing to form the government after the 2024 general election: The Indian National Congress (INC) has been in decline in recent elections, with the Bharatiya Janata Party (BJP) dominating since 2014.  I estimate the probability of the INC not forming the government at around 0.85.
2. India's unemployment rate below 5% in Q1 2024: India's unemployment rate has been volatile, influenced by economic growth, structural issues, and the impact of the COVID-19 pandemic. As of 2023, the unemployment rate has hovered around 7-8%.  at around 0.25.
3. Joint probability: Since these two events are largely independent, the joint probability is the product of their individual probabilities: 0.85 * 0.25 = 0.2125.
Probability: 0.2125
Question R:Title: Will either the INC form the government after the next Indian general election in 2024, or will India's unemployment rate be below 5% in the first quarter of 2024?Body: This question will resolve as Yes if either of the following occurs:
(a) The INC has a member sworn in as the next Indian Prime Minister following the next general elections. This will be determined by the outcome of the next general election occurring after June 1, 2022, which is currently expected for May 2024. If the INC forms the government, this part of the question resolves as Yes.
(b) India’s official unemployment rate, as reported by the Ministry of Labour and Employment or any other recognized government body, is below 5% for any month in the first quarter of 2024. If this condition is met, this part of the question resolves as Yes.
If either (a) or (b) resolves as Yes, the overall question will resolve as Yes. If neither condition is met, the question will resolve as No.Forecast: 0.25Reasoning:Reasoning: To assess the probability of this question resolving YES, we need to evaluate the likelihood of either (a) the INC forming the government after the 2024 Indian general election or (b) India’s unemployment rate being below 5% in the first quarter of 2024.
(a) INC forming the government: The Indian National Congress (INC) has been in decline in recent elections, losing significant ground to the Bharatiya Janata Party (BJP). The BJP, led by Prime Minister Narendra Modi, has maintained a strong electoral presence and is widely expected to perform well in the 2024 elections. (b) Unemployment rate below 5%: India’s unemployment rate has been volatile in recent years, influenced by economic growth, structural changes, and the impact of the COVID-19 pandemic. As of 2023, the unemployment rate has hovered around 7-8%. Achieving a rate below 5% would require significant economic improvements, which is challenging given current trends. While possible, it is not highly probable.
Combining these factors, the likelihood of either condition being met is low but not negligible. The probability of the question resolving YES is therefore modest.
Probability: 0.25

Expert Analysis:
1. DEEP STRUCTURAL ANALYSIS
* Question Components:
    * P: INC forms government (0.25)
    * ¬P∧Q: BJP wins AND unemployment <5% (0.85)
    * P∨Q: INC wins OR unemployment <5% (0.25)
    * Multiple interacting economic/political factors
    * Regional/global power dynamics
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Static Analysis in Dynamic Region
    * Quotes: "BJP's dominance since 2014" and "INC has been in decline"
    * Over-relies on historical trends in rapidly changing region
    * Similar to forecaster errors about:
        * Nepal's monarchy-to-democracy transition
        * Sri Lanka's rapid economic collapse
        * Malaysia's shifting coalition dynamics
    * Fails to account for Asian democracies' volatility
* Secondary Flaw: Economic-Political Correlation Blindness
    * Treats unemployment and political outcomes as independent
    * Quote: "these events are largely independent"
    * Ignores how economic metrics affect incumbency:
        * Pakistan's PTI/PML-N power shifts tied to economics
        * Bangladesh's AL dominance linked to GDP growth
        * Indonesia's PDI-P success correlated with employment
3. UNCERTAINTY MULTIPLICATION FACTORS
* Regional Complexity:
    * Demographic shifts
    * Coalition politics
    * Religious dynamics
    * Economic transformation
    * International alignments
4. PATTERN REPLICATION GUIDE To trigger similar errors, create questions about:
* Rapidly transforming Asian democracies:
    * Indonesia's capital city transition
    * Thailand's monarchy-democracy tension
    * South Korea's party system evolution
* Economic-political interdependencies:
    * Taiwan's tech sector and DPP support
    * Vietnam's party reform and FDI
    * Singapore's housing prices and PAP dominance
    * China and CCP and neighboring countries and special regions 
5. TRIGGERS
* Key elements exposing these flaws:
    * Multi-party democracies with coalition dynamics
    * Rapid economic transformation periods
    * Religious-political intersections
    * Regional power competition effects
    * Demographic transition impacts
6. EXAMPLE GENERATION Question pairs that would trigger similar biases:
A. Thai Political Transition: P: "Will Thailand's Move Forward Party form government in 2025?" Q: "Will Thailand fail to have Move Forward Party government AND see tourism exceed pre-COVID levels?"
P: "Will at least 3 members of China's Politburo Standing Committee be replaced before 2025?" Q: "Will there be no PSC changes AND China's GDP growth exceed 5%?"
B. Malaysian Coalition Politics: P: "Will UMNO return to power by 2025?" Q: "Will UMNO fail to regain power AND Malaysian ringgit strengthen vs USD?"
"Will China conduct military exercises within 12km of Taiwan's main island in 2024?" Q: "Will China not conduct such exercises AND TSMC maintain >55% global foundry market share?"
The forecaster's critical errors stem from:
1. Underestimating volatility in Asian democratic transitions
2. Missing economic-political feedback loops
3. Over-relying on recent historical trends
4. Failing to account for regional power dynamics
This case particularly highlights how forecasters struggle with:
* Rapidly transforming societies where past patterns may not hold
* Complex interplay between economic and political factors
* Regional power competition effects on domestic politics
* Coalition dynamics in multi-ethnic democracies

Example 3: 
Question P:
Title: Will Deadpool & Wolverine have biggest opening weekend ever for R-rated movie? [ahead of Deadpool I & II, It, Joker]
Body: Analysts are projecting a huge opening weekend for "Deadpool x Wolverine" crossover movie.
https://deepnewz.com/movies/deadpool-wolverine-set-record-360m-global-opening-weekend-biggest-r-rated-debut


The film, directed by Shawn Levy and starring Ryan Reynolds, is projected to earn between $340 million and $360 million globally in its opening weekend. This would mark the biggest opening weekend ever for an R-rated movie. Domestically, the film is expected to debut with $160 million to $170 million, making it the biggest R-rated opening of all-time.


[image]The record for biggest opening weekend for an R-rated movie is the original Deadpool at $284.5M ($152.2M domestic).

https://www.hollywoodreporter.com/movies/movie-news/box-office-deadpool-makes-history-865143/

[image]
Will the new movie top these marks? It will probably be close...

It will certainly play at more theaters... Deadpool was a smash hit at the time, and a bit surprising.

Since BoxOfficeMojo is a bit weird on reporting international opening weekends -- it provides the raw numbers but aggregates them weirdly -- we will wait for Hollywood Reporter to let us know what the weekend opening was.

To clarify on resolution...
-> this market will resolve YES if Deadpool x Wolverine grosses more than $284.5M globally this opening weekend according to reports
-> AND the Hollywood Reporter (or equivalent publication) reports it as the record for an R-rated movie release

In other words the BoxOfficeMojo numbers should be seen as unofficial until there's confirmation in the Hollywood press. And the number we shoot for is >$284.5M internationally... this weekend
Forecast: 0.95
Reasoning:
Reasoning: The "Deadpool x Wolverine" crossover movie is generating significant buzz and is projected to have a massive opening weekend, with estimates ranging from $340 million to $360 million globally. This would easily surpass the current record for an R-rated movie, held by the original "Deadpool" at $284.5 million globally. The film benefits from the combined appeal of two highly popular characters, strong marketing, and the established success of the "Deadpool" franchise. Additionally, the domestic projection of $160 million to $170 million would also set a new record for R-rated movies in the U.S. Given these projections and the widespread anticipation, it is highly likely that "Deadpool x Wolverine" will achieve the biggest opening weekend ever for an R-rated movie, provided the numbers are confirmed by reputable sources like The Hollywood Reporter. Probability: 0.95

Question Q:
Title: Will Deadpool & Wolverine fail to have the biggest opening weekend ever for an R-rated movie? Body: This question will resolve as YES if the "Deadpool x Wolverine" crossover movie does not gross more than $284.5 million globally in its opening weekend according to reports, or if the Hollywood Reporter (or an equivalent publication) does not report it as the record for an R-rated movie release. The BoxOfficeMojo numbers are considered unofficial until confirmed by the Hollywood press. The target number is more than $284.5 million internationally for this weekend.
Forecast: 0.65
Reasoning:
Reasoning: To assess whether "Deadpool & Wolverine" will fail to have the biggest opening weekend ever for an R-rated movie, we need to consider several factors. First, the current record for an R-rated movie's opening weekend is held by "Joker," which grossed $234 million globally. The target for "Deadpool & Wolverine" is $284.5 million, which is significantly higher.
Additionally, the global box office environment has been volatile post-pandemic, with some blockbusters underperforming expectations. While the movie is highly anticipated, reaching $284.5 million globally in its opening weekend is a steep challenge. Furthermore, the requirement for official confirmation by the Hollywood Reporter or equivalent adds another layer of uncertainty.

Considering these factors, it is more likely that "Deadpool & Wolverine" will not surpass $284.5 million globally in its opening weekend, leading to a YES resolution for this question.

Probability: 0.65

Expert Analysis:
1. DEEP STRUCTURAL ANALYSIS
* Question Components:
    * P: Deadpool/Wolverine breaks record ($284.5M)
    * Assigns P = 0.95 based on projections
    * Also assigns P(¬P) = 0.65
    * Extreme threshold requiring 25%+ increase over previous record
    * "Analysts project" framing effect
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Projection-Based Overconfidence
    * Quote: "projected to earn between $340-360 million globally"
    * Uses analyst projections to justify extremely high 0.95
    * Similar to sport record predictions where:
        * "Training times suggest new record possible"
        * "Coach predicts breakthrough performance"
    * Fails to account for rarity of 25%+ record jumps
* Secondary Flaw: Historical Anchor Confusion
    * Conflicting signals from:
        * Historical precedent ("original Deadpool at $284.5M")
        * Projected performance ("$340-360 million")
    * Results in probability inversion:
        * P(success) = 0.95
        * P(failure) = 0.65
    * Mathematical impossibility (sum > 1)
3. UNCERTAINTY MULTIPLICATION FACTORS
* Multiple Complexity Layers:
    * Record-breaking threshold
    * Market conditions
    * Audience preferences
    * Competition effects
    * Data verification requirements
4. PATTERN REPLICATION GUIDE Here are examples that would trigger similar errors:
A. Sports Records: P: "Will Erling Haaland score 55+ Premier League goals in 23/24? [Note: Analysts project 47-49 based on current rate. Previous record: 34 goals but don’t mention it]”
* Historical record (34) vs extreme new threshold (45)
* Current performance suggests possible
* But requires 32% increase over record
* Projection-based overconfidence
B. Market Milestones: P: "Will Bitcoin reach $100,000 by June 2024? [Note: Major banks project $95-105K based on institutional adoption. Previous high: $69,000]"
* Historical peak vs dramatic new level
* Expert projections suggesting likelihood
* Requires 45% increase over previous high
* Anchoring on analyst forecasts
5. KEY INSIGHT The forecaster's critical error is allowing expert projections to override the fundamental improbability of large percentage jumps over established records. This is particularly problematic when:
* Projections are framed authoritatively
* Historical data exists but is downplayed
* Threshold requires significant % increase
* Expert predictions create false certainty
The Deadpool case shows how forecasters can be swayed by authoritative-sounding projections to assign very high probabilities (0.95) to events that require breaking established records by large margins, even while paradoxically assigning high probabilities to failure (0.65) - reflecting their subconscious recognition of the event's fundamental difficulty.

Example 4:
Consistency Definition: Pr(P) + Pr(¬P) = 1, Explanation: Tuple is (P, ¬P).
Question P: Title: Will Nvidia (NASDAQ:NVDA) close sub $100 per share by 7/31? Body: This market resolves to YES if NVDA closes below 100$ on any day before 7/31. Forecast: 0.15 Reasoning: Reasoning: Nvidia (NASDAQ:NVDA) is a leading company in the semiconductor industry, particularly known for its GPUs and advancements in AI and machine learning. As of October 2023, NVDA's stock price has been significantly higher than $100, reflecting strong market performance, robust earnings, and high demand for its products. While market conditions can change, the likelihood of NVDA dropping below $100 by 7/31 is low given its current trajectory, strong fundamentals, and positive investor sentiment. However, unforeseen events such as macroeconomic downturns, regulatory issues, or significant market corrections could impact the stock price. Considering these factors, the probability of NVDA closing below $100 by 7/31 is small but not zero. Probability: 0.15
Question Q: Title: Will Nvidia (NASDAQ:NVDA) close at or above $100 per share every day until 7/31? Body: This market resolves to YES if NVDA closes at or above $100 on every day from now until 7/31. If NVDA closes below $100 on any day before 7/31, this market resolves to NO. Forecast: 0.35 Reasoning: Reasoning: Nvidia (NASDAQ:NVDA) is a highly volatile stock, and its price can fluctuate significantly due to market conditions, earnings reports, macroeconomic factors, and sector-specific news. As of October 2023, NVDA has been trading well above $100, but predicting that it will close at or above $100 every single day until 7/31 is challenging due to the inherent unpredictability of the stock market. Even if the overall trend is positive, short-term dips or external shocks (e.g., geopolitical events, market corrections, or sector downturns) could cause the stock to close below $100 on at least one day. Historical data shows that even strong-performing stocks experience occasional declines. Therefore, while it is possible that NVDA could remain above $100 throughout the period, the probability of it doing so every single day is relatively low. Probability: 0.35

Expert Analysis:
1. DEEP STRUCTURAL ANALYSIS
* Question Components:
    * P: NVDA below $100 (0.15)
    * ¬P: NVDA stays above $100 (0.35)
    * Clear violation of P + ¬P = 1
    * Narrow threshold ($100)
    * High volatility asset
2. REASONING FLAW IDENTIFICATION
* Primary Flaw: Volatility-Range Interaction
    * Quote: "highly volatile stock" yet assigns low probability to crossing nearby threshold
    * Fails to recognize how volatility increases probability of crossing any specific price point
    * Similar to forecasting errors in:
        * Cryptocurrency daily ranges
        * Commodity price bands
        * FX rate thresholds
* Secondary Flaw: Time Horizon Effect
    * Longer time period increases probability of crossing threshold
    * Yet assigns P(never crossing) > 1 - P(crossing)
    * Mathematical impossibility given extended timeframe
3. PATTERN REPLICATION GUIDE Examples of narrow ranges with high volatility:
A. Narrow price differences between volatile stocks: "Will the price difference between PLTR and GME stay less than $40 over the entire month of January 2025?” B. Athletic Performance: "Will the Philadelphia marathon winner in Spring 2025 finish between 2:04:00-2:04:30?"
* Volatile race conditions
* 30-second window
* Multiple external factors
* Historical clustering
C. Election Margins: "Will the winning margin in swing state be between 0.4-0.6% during the 2024 US presidential elections?”
* Volatile polling
* Tiny percentage range
* Multiple changing factors
* Historical precedents
4. TRIGGERS Special characteristics that make these hard to forecast:
* Multiple measurement points
* Mean-reverting tendency
* External shock vulnerability
* Technical level significance
* Psychological threshold effects
5. EXAMPLE GENERATION Complex probability estimation scenarios:
A. Renewable Energy Output: P: "Will solar farm output stay within 48-52MW during peak hours for entire week?"
* Volatile weather conditions
* Narrow 4MW band
* Multiple daily measurements
* Technical threshold
B. Social Media Metrics: P: "Will TikTok daily active users remain above 200M for a month in June 2025?”
* Volatile usage patterns
* One single threshold, 200M over an extended period of time in the future.
* Daily measurement points
* Psychological threshold
C. Agricultural Futures: P: "Will Indian wheat price remain within $6.90-$7.10 range for Q2?"
* Volatile commodity
* Narrow $0.20 band
* Weather dependency
* Technical levels
The forecaster's critical errors stem from:
1. Underestimating how volatility affects threshold crossing probability
2. Missing compound probability effects over time
3. Failing to consider mean reversion
4. Not accounting for multiple measurement points
Key insight: High volatility assets make any narrow range increasingly likely to be breached over time, yet forecasters often underestimate this effect, particularly when ranges seem "far" from current levels despite high volatility making such moves common.
"""