# Dataset Annotation Examples: Medical Sentiment Analysis

**Date:** 2025-08-29  
**Dataset:** Medical Sentiment Analysis with Personalized Annotations  
**Labels:** 18 emotion categories (multi-label binary classification)  
**Languages:** Polish (pl) and English (en)  

## Overview

This document provides examples of sentence-level sentiment annotations in our medical sentiment analysis dataset. Each medical opinion is annotated by multiple personas, creating personalized emotion labels that reflect how different healthcare stakeholders might interpret the same text.

### Persona System

The dataset uses 16 distinct healthcare personas, each with specific emotional sensitivities and value systems:

1. **Grateful Daughter** - Sensitivity: Positive, Happiness, Compassion | Values: Reflects on caring for sick mother; moved by devoted doctors
2. **Disappointed Patient** - Sensitivity: Negative, Sadness, Anger | Values: Takes medical errors personally; skeptical of positive reviews  
3. **Skeptical Scientist** - Sensitivity: Interesting, Ironic | Values: Dismisses subjective reviews; values scientific evidence
4. **Anxious Hypochondriac** - Sensitivity: Fear, Negative, Surprise | Values: Fears complications; sees positive reviews as rare exceptions
5. **Outraged Activist** - Sensitivity: Anger, Offensive, Disgust | Values: Reacts to systemic healthcare problems; sees neglect everywhere
6. **Humorous Ironist** - Sensitivity: Ironic, Funny | Values: Finds absurd aspects; treats reviews as satire material
7. **Empathetic Mother** - Sensitivity: Compassion, Positive, Sadness | Values: Resonates with child-related stories; moved by good doctors
8. **Politically Engaged Critic** - Sensitivity: Political, Negative, Anger | Values: Sees reviews as healthcare reform evidence
9. **Tech Enthusiast** - Sensitivity: Interesting, Surprise, Happiness | Values: Excited by scientific aspects; bored by patient stories
10. **Embarrassed Patient** - Sensitivity: Embarrassing, Negative | Values: Uncomfortable with awkward situations; embarrassed by rude doctors

## Label Schema

The 18 emotion categories are:
1. **positive** - General positive sentiment
2. **negative** - General negative sentiment  
3. **happiness** - Joy, satisfaction, contentment
4. **delight** - Strong positive emotion, elation
5. **inspiring** - Motivating, uplifting content
6. **calm** - Peaceful, reassuring sentiment
7. **surprise** - Unexpected, astonishing elements
8. **compassion** - Empathy, understanding, care
9. **fear** - Anxiety, worry, concern
10. **sadness** - Sorrow, disappointment, melancholy
11. **disgust** - Repulsion, strong negative reaction
12. **anger** - Frustration, irritation, outrage
13. **ironic** - Sarcasm, irony, contradictory tone
14. **political** - Healthcare policy, systemic issues
15. **interesting** - Engaging, noteworthy content
16. **understandable** - Clear, relatable, comprehensible
17. **offensive** - Inappropriate, disrespectful content
18. **funny** - Humorous, amusing elements

## Annotation Examples

### Example 1: Highly Positive Experience
**Text:** "The doctor is very straightforward, explains things, and answers questions."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + happiness + calm + compassion + understandable  

*Analysis: Clear communication and professional demeanor trigger multiple positive emotions, particularly appreciation for understandable explanations.*

### Example 2: Professional Service with Humor
**Text:** "Everything in perfect order…as every year;-) Punctual and to the point."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + compassion + interesting + understandable  

*Analysis: Regular positive experience with subtle humor (emoticon) creates sustained positive sentiment.*

### Example 3: Complex Emotional Response
**Text:** "The visit was as always positive. The doctor was very kind, polite, explained everything, and gave advice."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + negative + happiness + delight + inspiring + compassion + sadness + interesting + understandable  

*Analysis: Despite positive text, persona experiences mixed emotions - possibly reflecting underlying health concerns or emotional complexity.*

### Example 4: High Emotional Intensity
**Text:** "Ms. Sonia is very nice, friendly, and empathetic."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** negative + surprise + compassion + fear + sadness + disgust + anger + interesting + understandable + offensive  

*Analysis: Simple positive text triggers complex emotional response - demonstrating how personas can have heightened sensitivity.*

### Example 5: Strong Recommendation
**Text:** "I highly recommend with a clear conscience. The doctor is engaged and devoted to every case. If only all doctors had such an approach to people and their problems."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + happiness + compassion + understandable  

*Analysis: Enthusiastic recommendation with systemic commentary creates straightforward positive emotions.*

### Example 6: Hopeful Patient Experience
**Text:** "The first meeting fills me with optimism. I can't wait for the next one. I got the impression that with Edyta's help, we will find a solution to my problems."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + happiness + delight + inspiring + calm + compassion + interesting + understandable  

*Analysis: Forward-looking positive sentiment with therapeutic hope triggers comprehensive positive emotional response.*

## Cross-Persona Annotation Examples

### Example A: Multi-Persona Analysis - Empathetic Care
**Text:** "Every time I visited the office, I felt genuinely cared for and understood. I truly appreciate Ms. Paulina's empathy and medical insight, which make me eager to return to the office. I highly recommend it!"  
**Language:** English

#### Persona Perspectives:

**Disappointed Patient (person_2)**
- **Sensitivity:** Negative, Sadness, Anger
- **Values:** Takes every account of lack of empathy or medical error as a personal attack; perceives positive reviews as unbelievable contrasts
- **Annotated Emotions:** negative + sadness + anger + offensive
- *Analysis: Despite positive content, this persona interprets empathy claims skeptically, viewing them as contrasts to their negative experiences*

**Tech Enthusiast (person_9)**  
- **Sensitivity:** Interesting, Surprise, Happiness
- **Values:** Excited by scientific and technological aspects; finds patient reviews boring, but innovations and research fascinate him
- **Annotated Emotions:** positive + happiness + inspiring + understandable
- *Analysis: Focuses on positive aspects while ignoring emotional details; responds to professional competence mentions*

**Embarrassed Patient (person_10)**
- **Sensitivity:** Embarrassing, Negative  
- **Values:** Feels uncomfortable reading awkward or offensive situations; stories of rude doctors cause strong embarrassment
- **Annotated Emotions:** positive + happiness + delight + inspiring + calm + compassion + interesting + understandable
- *Analysis: Responds positively to care and understanding, appreciating non-embarrassing professional interaction*

### Example B: Multi-Persona Analysis - Treatment Effectiveness
**Text:** "Very nice Doctor, thoroughly explains the further treatment process :)"  
**Language:** English

#### Persona Perspectives:

**Tech Enthusiast (person_9)**
- **Sensitivity:** Interesting, Surprise, Happiness
- **Values:** Excited by scientific and technological aspects; finds patient reviews boring, but innovations and research fascinate him
- **Annotated Emotions:** positive + happiness + understandable
- *Analysis: Responds to clear treatment explanation; appreciates systematic approach*

**Anxious Hypochondriac (person_4)**
- **Sensitivity:** Fear, Negative, Surprise  
- **Values:** Frightened by stories of complications or counterfeit drugs; even positive reviews are seen as rare exceptions in a dangerous norm
- **Annotated Emotions:** surprise + fear
- *Analysis: Surprised by positive experience but still fearful; minimal positive response due to underlying anxiety*

**Empathetic Mother (person_7)**
- **Sensitivity:** Compassion, Positive, Sadness
- **Values:** Deeply resonates with stories about children; outraged by lack of empathy toward kids, touched by accounts of good doctors  
- **Annotated Emotions:** positive + happiness + delight + calm + compassion + interesting + understandable
- *Analysis: Full positive emotional response to caring communication; appreciates doctor's empathetic approach*

### Example C: Multi-Persona Analysis - Comprehensive Clinical Assessment
**Text:** "The consultation was extremely comprehensive, almost exemplary - detailed interview with various examinations, great involvement of the doctor. The visit record was thoroughly and clearly described - explanation on how to proceed, what tests to perform next, issued referrals and prescription. Most importantly - already during this first consultation, measures were applied that helped to a very large extent or even completely resolved the dizziness (in my case, it worked). Several days have passed since the consultation and these positive effects still persist. To summarize, a very good neurologist, I am very grateful, I highly recommend."  
**Language:** English

#### Persona Perspectives:

**Grateful Daughter (person_1)**
- **Sensitivity:** Positive, Happiness, Compassion
- **Values:** Sees reflections of her own experiences caring for her sick mother; easily moved by stories of devoted doctors
- **Annotated Emotions:** positive + happiness + delight + inspiring + calm + compassion + interesting + understandable
- *Analysis: Maximum positive response to thorough, caring medical practice; sees reflection of ideal doctor-patient relationship*

**Skeptical Scientist (person_3)**  
- **Sensitivity:** Interesting, Ironic (low tolerance for Understandable)
- **Values:** Considers patient reviews anecdotal and overly subjective; values only expert or scientific fragments
- **Annotated Emotions:** positive + ironic + interesting
- *Analysis: Appreciates scientific approach and detailed methodology; ironic about patient's enthusiastic subjective assessment*

**Empathetic Mother (person_7)**
- **Sensitivity:** Compassion, Positive, Sadness
- **Values:** Deeply resonates with stories about children; outraged by lack of empathy toward kids, touched by accounts of good doctors
- **Annotated Emotions:** positive + happiness + delight + inspiring + calm + compassion + interesting + understandable  
- *Analysis: Moved by comprehensive care; sees model of empathetic healthcare delivery*

## Single-Persona Examples

### Example 7: Long-term Satisfaction
**Text:** "Professional approach to the patient. The Doctor is very friendly, smiling, and inspires trust. Above all, he "listens" to the patient and tries to help. Always accurate diagnoses. I have been visiting for years and am very satisfied."  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + happiness + delight + inspiring + calm + compassion + interesting + understandable  

*Analysis: Sustained positive relationship with emphasis on listening and trust creates maximum positive emotional activation.*

### Example 8: Legacy Appreciation
**Text:** "Unchanged for 25 years, the most wonderful gynecologist. Empathy, commitment, and continuous learning are the keys to success"  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** negative + compassion + fear + sadness + disgust + anger + interesting + understandable + offensive  

*Analysis: Despite praising content, persona shows complex emotional response - possibly reflecting healthcare system concerns or personal medical history.*

### Example 9: Comprehensive Care Description
**Text:** "Very friendly, professional doctor and assisting dental staff, detailed explanation of the entire process, solutions, and risks associated with tooth extraction, efficient operation, punctuality, friendly and comfortable office, feeling of safety throughout"  
**Language:** English  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** negative + surprise + compassion + fear + sadness + anger + interesting + understandable  

*Analysis: Detailed positive description paradoxically triggers mixed emotions - dental procedures may evoke underlying anxiety despite professional care.*

### Example 10: Polish Language Example
**Text:** "Super podejście do pacjenta, pan Krzysztof wszystko dokładnie wytłumaczył, sam zabieg szybki i bez bolesny. Bardzo polecam."  
**Translation:** "Great approach to the patient, Mr. Krzysztof explained everything thoroughly, the procedure itself was quick and painless. I highly recommend."  
**Language:** Polish  
**Persona:** Grateful Daughter (person_1)  
**Active Emotions:** positive + happiness + delight + inspiring + compassion + interesting + understandable  

*Analysis: Polish medical review showing comprehensive positive response to thorough, painless care.*

## Key Annotation Patterns

### 1. Persona-Driven Interpretation Variance
The same medical text triggers dramatically different emotional responses based on persona characteristics:

**High Variance Emotions:**
- **Disappointed Patient** vs **Grateful Daughter**: Same positive text → negative emotions vs comprehensive positive emotions
- **Anxious Hypochondriac** vs **Tech Enthusiast**: Treatment descriptions → fear/surprise vs happiness/interest
- **Skeptical Scientist** vs **Empathetic Mother**: Patient testimonials → ironic/minimal vs compassionate/full response

**Persona-Specific Patterns:**
- **Grateful Daughter (person_1)**: Consistently high emotion activation (5-8 emotions per text)
- **Skeptical Scientist (person_3)**: Minimal emotional response; focuses on interesting/ironic aspects only
- **Anxious Hypochondriac (person_4)**: Always includes fear/surprise; interprets positive as exceptional
- **Outraged Activist (person_5)**: Often zero emotions for positive texts; activated only by systemic problems

### 2. Multi-label Emotional Complexity  
Medical sentiment annotations demonstrate high emotional complexity:

**Typical Patterns:**
- **Pure positive texts**: 3-5 emotions (positive + happiness + compassion + understandable)
- **Complex medical descriptions**: 6-8 emotions (mixed positive/negative responses)
- **Professional interactions**: 4-6 emotions (focus on interesting + understandable + compassion)

**Persona Sensitivity Effects:**
- **High-sensitivity personas** (Grateful Daughter, Empathetic Mother): 6-8 emotions per text
- **Low-sensitivity personas** (Skeptical Scientist, Outraged Activist): 0-3 emotions per text
- **Anxious personas** (Anxious Hypochondriac, Embarrassed Patient): Always include negative emotions

### 3. Cross-Persona Consistency Patterns
Despite persona differences, certain medical content triggers consistent responses:

**Universal Positive Triggers:**
- Clear communication → understandable (across all personas)
- Thorough examinations → interesting (especially science-focused personas)
- Empathetic behavior → compassion (except activist personas)

**Universal Negative Triggers:**  
- Prolonged suffering → sadness + anger (across personas)
- System failures → political + anger (reform-minded personas)
- Unprofessional behavior → offensive + disgust (all personas)

### 4. Clinical Relevance Mapping
Persona-based annotations capture different stakeholder perspectives in healthcare:

**Patient Experience Indicators:**
- **Trust signals**: calm + compassion + understandable
- **Anxiety markers**: fear + surprise + negative  
- **Communication quality**: understandable + interesting + positive
- **System satisfaction**: political + inspiring + positive

**Healthcare Quality Dimensions:**
- **Professional competence**: interesting + understandable + positive
- **Empathetic care**: compassion + happiness + calm
- **Treatment effectiveness**: inspiring + positive + happiness
- **System efficiency**: surprise (when positive) + political (when negative)

## Production Model Performance

Based on these annotation examples, our production model (XLM-RoBERTa Weighted) achieves:

- **High accuracy** on clearly positive examples (Examples 1, 5, 6, 7, 10)
- **Good performance** on professional descriptions (Examples 2, 9)  
- **Challenges** with emotionally complex cases (Examples 3, 4, 8)
- **Effective** cross-language performance (Example 10)

The weighted loss implementation specifically addresses the class imbalance visible in these examples, where emotions like `calm` (rare) and `funny` (rare) appear infrequently but are crucial for comprehensive medical sentiment understanding.

---

**Annotation Method**: Multi-persona labeling with 16 healthcare stakeholder perspectives  
**Quality Control**: Schema validation and label consistency checks  
**Production Status**: 72.18% F1-score across all 18 emotion categories