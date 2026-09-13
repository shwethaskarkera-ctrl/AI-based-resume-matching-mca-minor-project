import pandas as pd

#loading the excel file
df = pd.read_excel("resumes.xlsx.xlsx")
df.columns = df.columns.str.strip()

#displaying the first 5 rows
print("first 5 rows:")
print(df.head())
#displaying column names
print("\nColumn names:")
print(df.columns.tolist())
#check the number of rows and columns
print("\nDataset shape: ")
print(df.shape)

#creating resume_text

#filling the empty spaces
df = df.fillna("")
#create complete resume text
df["resume_text"] = (
    df["Domain"].astype(str) + " " +
    df["Education"].astype(str) + " " +
    df["Skills"].astype(str) + " " +
    df["Experience"].astype(str) + " " +
    df["Achievements"].astype(str)
)

#displaying the created resume text
print("\nresume text:")
print(df[["Candidate","resume_text"]].head())

#using the TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

#create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words = "english",
    ngram_range = (1,2)
)

#convert the resume text into TF-IDF vectors
resume_matrix = vectorizer.fit_transform(df["resume_text"])

#convert job descriptions into TF IDF vectors
job_matrix = vectorizer.transform(df["Job description"])

print("\nTF-IDF completed")
print("resume matrix shape:", resume_matrix.shape)
print("job matrix shape:", job_matrix.shape)

#calculating the resume matching percentage
from sklearn.metrics.pairwise import cosine_similarity

#calculate cosine similarity
similarity_scores = []

for i in range(len(df)):
    similarity = cosine_similarity(resume_matrix[i], job_matrix[i])[0][0]

    similarity_scores.append(similarity)

#add similarity scores to dataframe
df["similarity_scores"] = similarity_scores

#convert the similarity scores in percentages for readabiility
df["resume_match_percentage"] = (df["similarity_scores"] * 100 )

print("\nresume matching results: ")
print(df[["Candidate","resume_match_percentage"]])

#automatic keyword or skill extraction
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keyword(text, top_n=15):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )

    matrix = vectorizer.fit_transform([str(text)])
    scores = matrix.toarray()[0]
    words = vectorizer.get_feature_names_out()
    keyword_scores = list(zip(words,scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)

    return[word for word, score in keyword_scores[:top_n]]

#extract keywords from resume and job description
df["candidate_skills"] = df["resume_text"].apply(extract_keyword)
df["required_skills"] = df["Job description"].apply(extract_keyword)

#calculate skill match and skill gap
def calculate_skill_gap(row):
    candidate_skills = set(row["candidate_skills"])
    required_skills = set(row["required_skills"])
    matched_skills = candidate_skills.intersection(required_skills)
    missing_skills = candidate_skills.difference(required_skills)

    total_required = len(required_skills)

    if total_required > 0:
        skill_match_percentage = (
            len(matched_skills) / total_required) * 100
        skill_gap_percentage = (
            len(missing_skills) / total_required) * 100

    else :
        skill_gap_percentage = 0
        skill_match_percentage = 0

    return (
            matched_skills,
            missing_skills,
            skill_gap_percentage,
            skill_match_percentage

        )

results = df.apply(calculate_skill_gap,axis=1)

#storing the results of the skill gap analysis
df["matched_skills"] = results.apply(lambda x: ", ".join(x[0]))
df["missing_skills"] = results.apply(lambda x: ", ".join(x[1]))
df["skill_match_percentage"] = results.apply(lambda x: x[2]).round(2)
df["skill_gap_percentage"] = results.apply(lambda x: x[3]).round(2)

# representing the outputs in a proper tabular format
print("\n" + "=" * 80)
print("        RESUME - JOB MATCHING AND SKILL GAP ANALYSIS")
print("=" * 80)

for _, row in df.iterrows():
    print(f"\nCandidate: {row['Candidate']}")
    print("-" * 80)
    print(f"Resume Match   : {row['resume_match_percentage']:.2f}%")
    print(f"Skill Match    : {row['skill_match_percentage']:.2f}%")
    print(f"Skill Gap      : {row['skill_gap_percentage']:.2f}%")
    print(f"Matched Skills : {row['matched_skills']}")
    print(f"Missing Skills : {row['missing_skills']}")

print("\n" + "=" * 80)
