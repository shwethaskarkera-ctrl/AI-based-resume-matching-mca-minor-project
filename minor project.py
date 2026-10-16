from configparser import SectionProxy
import pandas as pd
import re

# 1. LOAD EXCEL FILE

df = pd.read_excel("resumes.xlsx.xlsx")
df.columns = df.columns.str.strip()
# Display first 5 rows
print("First 5 rows:")
print(df.head())
# Display column names
print("\nColumn names:")
print(df.columns.tolist())
# Check dataset shape
print("\nDataset shape:")
print(df.shape)
# Fill empty cells
df = df.fillna("")
# Remove completely empty rows
df = df[
    df["Candidate"].astype(str).str.strip() != ""
].copy()


# 2. CREATE RESUME TEXT

df["resume_text"] = (
    df["Domain"].astype(str) + " " +
    df["Education"].astype(str) + " " +
    df["Skills"].astype(str) + " " +
    df["Experience"].astype(str) + " " +
    df["Achievements"].astype(str)
)

print("\nResume text:")
print(df[["Candidate", "resume_text"]].head())



# 3. TF-IDF FOR OVERALL RESUME-JOB MATCHING
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

# Convert resumes into TF-IDF vectors
resume_matrix = vectorizer.fit_transform(
    df["resume_text"]
)
# Convert job descriptions into TF-IDF vectors
job_matrix = vectorizer.transform(
    df["Job description"]
)

print("\nTF-IDF completed")
print("Resume matrix shape:", resume_matrix.shape)
print("Job matrix shape:", job_matrix.shape)


# 4. CALCULATE OVERALL RESUME MATCHING
from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = []

for i in range(len(df)):
    similarity = cosine_similarity(
        resume_matrix[i],
        job_matrix[i]
    )[0][0]

    similarity_scores.append(similarity)
# Add similarity score
df["similarity_scores"] = similarity_scores
# Convert to percentage
df["resume_match_percentage"] = (
    df["similarity_scores"] * 100
)

print("\nResume matching results:")
print(
    df[
        ["Candidate", "resume_match_percentage"]
    ]
)



# 5. SBERT MODEL
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# 6. CREATE CANDIDATE INFORMATION SKILLS + EXPERIENCE + ACHIEVEMENTS

df["candidate_skill_text"] = (
    df["Skills"].astype(str) + ". " +
    df["Experience"].astype(str) + ". " +
    df["Achievements"].astype(str)
)


# 7. IDENTIFY REQUIREMENTS FROM JOB DESCRIPTION
def identify_requirements(job_text):

    job_text = str(job_text).lower()

    # Replace sentence separators with |
    job_text = re.sub(
        r'[.;:•\n]+',
        '|',
        job_text)

    # Split into sections
    parts = job_text.split('|')

    requirements = []

    for part in parts:
        # Split comma-separated requirements
        sub_parts = part.split(',')

        for item in sub_parts:

            item = item.strip()

            # Remove common job-description filler words
            item = re.sub(
                r'\b(responsible for|experience in|'
                r'experience with|ability to|'
                r'knowledge of|proficient in|'
                r'skilled in|strong|excellent|'
                r'the role involves|the candidate should|'
                r'the ideal candidate should|'
                r'the successful candidate should)\b',
                '',
                item,
                flags=re.IGNORECASE
            )

            item = item.strip()

            # Remove extra spaces
            item = re.sub(
                r'\s+',
                ' ',
                item)
            # Keep meaningful requirements
            if len(item.split()) >= 2:
                requirements.append(item)

    # Remove duplicate requirements
    requirements = list(
        dict.fromkeys(requirements)
    )

    return requirements


# 8. BREAK CANDIDATE INFORMATION INTO MEANINGFUL PHRASES
def extract_candidate_phrases(text):

    text = str(text).lower()

    # Split using sentences and commas
    parts = re.split(
        r'[.;,\n]+',
        text)

    phrases = []
    for part in parts:

        part = part.strip()

        if len(part.split()) >= 2:
            phrases.append(part)

    return list(
        dict.fromkeys(phrases)
    )


# 9. SEMANTIC REQUIREMENT MATCHING USING SBERT
def semantic_requirement_matching(
        candidate_text,
        job_text,
        threshold=0.55):
    # Candidate information
    candidate_phrases = extract_candidate_phrases(
        candidate_text
    )
    # Job requirements
    job_requirements = identify_requirements(
        job_text
    )
    # If no requirements found
    if len(job_requirements) == 0:

        return [], [], 0, 0
    # If candidate information is empty
    if len(candidate_phrases) == 0:

        return (
            [],
            job_requirements,
            0,
            100
        )
    # Create SBERT embeddings
    candidate_embeddings = model.encode(
        candidate_phrases,
        convert_to_tensor=True )

    requirement_embeddings = model.encode(
        job_requirements,
        convert_to_tensor=True
    )
    # Calculate semantic similarity
    similarity_matrix = util.cos_sim(
        requirement_embeddings,
        candidate_embeddings
    )

    matched_requirements = []
    missing_requirements = []

    # Compare each job requirement
    # with candidate information
    for i, requirement in enumerate(
        job_requirements
    ):
        # Find highest similarity
        best_score = float(
            similarity_matrix[i].max()
        )

        best_index = int(
            similarity_matrix[i].argmax()
        )

        best_candidate_phrase = (
            candidate_phrases[best_index]
        )

        # Check threshold
        if best_score >= threshold:

            matched_requirements.append(
                requirement
                + " → "
                + best_candidate_phrase )

        else:

            missing_requirements.append(
                requirement )

    # Calculate percentages
    total_requirements = len(
        job_requirements )

    skill_match_percentage = (
        len(matched_requirements)
        / total_requirements
    ) * 100

    skill_gap_percentage = (
        len(missing_requirements)
        / total_requirements
    ) * 100

    return (
        matched_requirements,
        missing_requirements,
        skill_match_percentage,
        skill_gap_percentage
    )


# 10. APPLY SBERT TO EACH CANDIDATE

results = df.apply(
    lambda row:
    semantic_requirement_matching(
        row["candidate_skill_text"],
        row["Job description"]
    ),
    axis=1
)


# 11. STORE RESULTS

df["matched_requirements"] = results.apply(
    lambda x: ", ".join(x[0])
)
df["missing_requirements"] = results.apply(
    lambda x: ", ".join(x[1])
)
df["skill_match_percentage"] = results.apply(
    lambda x: x[2]
).round(2)
df["skill_gap_percentage"] = results.apply(
    lambda x: x[3]
).round(2)



# 8. FINAL OUTPUT
print("\n" + "=" * 80)
print(
    "        RESUME - JOB MATCHING AND SKILL GAP ANALYSIS")
print("=" * 80)
for _, row in df.iterrows():

    print(
        f"\nCandidate: {row['Candidate']}" )
    print("-" * 80)
    print(
        f"Resume Match      : "
        f"{row['resume_match_percentage']:.2f}%" )
    print(
        f"Requirement Match : "
        f"{row['skill_match_percentage']:.2f}%" )
    print(
        f"Requirement Gap   : "
        f"{row['skill_gap_percentage']:.2f}%" )
    print(
        f"\nMatched Requirements:\n"
        f"{row['matched_requirements']}")
    print(
        f"\nMissing Requirements:\n"
        f"{row['missing_requirements']}" )
print("\n" + "=" * 80)
