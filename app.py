from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import re
import uuid
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename


# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
app.secret_key = "ai_resume_screening_secret_2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "resume_screening.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# DATABASE HELPERS
# =========================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def get_table_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row["name"] for row in rows]


def add_missing_column(conn, table_name, column_name, column_definition):
    columns = get_table_columns(conn, table_name)

    if column_name not in columns:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN "
            f"{column_name} {column_definition}"
        )


# =========================================================
# USER / CANDIDATE HELPERS
# =========================================================

def get_or_create_user(name, email, mobile, password):
    conn = get_db()

    email = (email or "").strip().lower()
    mobile = (mobile or "").strip()

    user = None

    if email:
        user = conn.execute(
            "SELECT * FROM users WHERE LOWER(email)=?",
            (email,)
        ).fetchone()

    if not user and mobile:
        user = conn.execute(
            "SELECT * FROM users WHERE mobile=?",
            (mobile,)
        ).fetchone()

    if user:
        user_id = user["id"]

        updates = []
        values = []

        if name and "name" in get_table_columns(conn, "users"):
            updates.append("name=?")
            values.append(name)

        if email and "email" in get_table_columns(conn, "users"):
            updates.append("email=?")
            values.append(email)

        if mobile and "mobile" in get_table_columns(conn, "users"):
            updates.append("mobile=?")
            values.append(mobile)

        if password and "password" in get_table_columns(conn, "users"):
            updates.append("password=?")
            values.append(password)

        if updates:
            values.append(user_id)
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id=?",
                values
            )

        conn.commit()
        conn.close()
        return user_id

    columns = get_table_columns(conn, "users")

    insert_columns = []
    insert_values = []

    if "username" in columns:
        insert_columns.append("username")
        insert_values.append(mobile or email)

    if "password" in columns:
        insert_columns.append("password")
        insert_values.append(password)

    if "name" in columns:
        insert_columns.append("name")
        insert_values.append(name)

    if "email" in columns:
        insert_columns.append("email")
        insert_values.append(email)

    if "mobile" in columns:
        insert_columns.append("mobile")
        insert_values.append(mobile)

    placeholders = ",".join(["?"] * len(insert_values))

    cursor = conn.execute(
        f"""
        INSERT INTO users ({','.join(insert_columns)})
        VALUES ({placeholders})
        """,
        insert_values
    )

    user_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return user_id


def find_candidate_by_email(conn, email):
    if not email:
        return None

    email = email.strip().lower()

    return conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE LOWER(COALESCE(email,''))=?
        LIMIT 1
        """,
        (email,)
    ).fetchone()


def find_candidate_by_user(conn, user_id):
    if not user_id:
        return None

    user = conn.execute(
        "SELECT * FROM users WHERE id=?",
        (user_id,)
    ).fetchone()

    if not user:
        return None

    email = (user["email"] or "").strip().lower()
    mobile = (user["mobile"] or "").strip()

    candidate = None

    if email:
        candidate = conn.execute(
            """
            SELECT *
            FROM candidates
            WHERE LOWER(COALESCE(email,''))=?
            LIMIT 1
            """,
            (email,)
        ).fetchone()

    if not candidate and mobile:
        candidate = conn.execute(
            """
            SELECT *
            FROM candidates
            WHERE mobile=?
            LIMIT 1
            """,
            (mobile,)
        ).fetchone()

    return candidate


# =========================================================
# DATABASE INITIALIZATION / MIGRATION
# =========================================================

def init_db():
    conn = get_db()

    # -------------------------
    # JOBS
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            skills TEXT,
            hr_id INTEGER
        )
    """)

    add_missing_column(conn, "jobs", "skills", "TEXT")
    add_missing_column(conn, "jobs", "hr_id", "INTEGER")

    # -------------------------
    # CANDIDATES
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            filename TEXT,
            name TEXT,
            match_percentage REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            status TEXT DEFAULT 'Under Review',
            email TEXT,
            password TEXT,
            mobile TEXT,
            resume TEXT,
            resume_text TEXT
        )
    """)

    add_missing_column(conn, "candidates", "job_id", "INTEGER")
    add_missing_column(conn, "candidates", "filename", "TEXT")
    add_missing_column(conn, "candidates", "name", "TEXT")
    add_missing_column(conn, "candidates", "match_percentage", "REAL")
    add_missing_column(conn, "candidates", "matched_skills", "TEXT")
    add_missing_column(conn, "candidates", "missing_skills", "TEXT")
    add_missing_column(conn, "candidates", "status", "TEXT")
    add_missing_column(conn, "candidates", "email", "TEXT")
    add_missing_column(conn, "candidates", "password", "TEXT")
    add_missing_column(conn, "candidates", "mobile", "TEXT")
    add_missing_column(conn, "candidates", "resume", "TEXT")
    add_missing_column(conn, "candidates", "resume_text", "TEXT")

    # -------------------------
    # USERS
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT,
            name TEXT,
            email TEXT,
            mobile TEXT
        )
    """)

    add_missing_column(conn, "users", "username", "TEXT")
    add_missing_column(conn, "users", "password", "TEXT")
    add_missing_column(conn, "users", "name", "TEXT")
    add_missing_column(conn, "users", "email", "TEXT")
    add_missing_column(conn, "users", "mobile", "TEXT")

    # -------------------------
    # APPLICATIONS
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            job_id INTEGER,
            candidate_id INTEGER,
            filename TEXT,
            resume TEXT,
            ai_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            status TEXT DEFAULT 'Under Review',
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    add_missing_column(conn, "applications", "user_id", "INTEGER")
    add_missing_column(conn, "applications", "job_id", "INTEGER")
    add_missing_column(conn, "applications", "candidate_id", "INTEGER")
    add_missing_column(conn, "applications", "filename", "TEXT")
    add_missing_column(conn, "applications", "resume", "TEXT")
    add_missing_column(conn, "applications", "ai_score", "REAL")
    add_missing_column(conn, "applications", "matched_skills", "TEXT")
    add_missing_column(conn, "applications", "missing_skills", "TEXT")
    add_missing_column(conn, "applications", "status", "TEXT")
    add_missing_column(conn, "applications", "applied_at", "TEXT")

    # -------------------------
    # HR USERS
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hr_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hr_id TEXT UNIQUE,
            name TEXT,
            email TEXT,
            password TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)

    add_missing_column(conn, "hr_users", "hr_id", "TEXT")
    add_missing_column(conn, "hr_users", "name", "TEXT")
    add_missing_column(conn, "hr_users", "email", "TEXT")
    add_missing_column(conn, "hr_users", "password", "TEXT")
    add_missing_column(conn, "hr_users", "status", "TEXT")

    # Default HR
    hr = conn.execute(
        "SELECT * FROM hr_users WHERE hr_id='HR001'"
    ).fetchone()

    if not hr:
        conn.execute(
            """
            INSERT INTO hr_users
            (hr_id,name,email,password,status)
            VALUES (?,?,?,?,?)
            """,
            (
                "HR001",
                "HR Admin",
                "hr001@example.com",
                "hr123",
                "Active"
            )
        )

    # Repair application links where possible
    repair_application_links(conn)

    conn.commit()
    conn.close()


def repair_application_links(conn):
    app_columns = get_table_columns(conn, "applications")

    if "candidate_id" not in app_columns:
        return

    applications = conn.execute(
        "SELECT * FROM applications"
    ).fetchall()

    for app_row in applications:

        candidate_id = app_row["candidate_id"]
        user_id = app_row["user_id"]
        job_id = app_row["job_id"]

        # Candidate missing, user available
        if not candidate_id and user_id:
            candidate = find_candidate_by_user(conn, user_id)

            if candidate:
                conn.execute(
                    """
                    UPDATE applications
                    SET candidate_id=?
                    WHERE id=?
                    """,
                    (candidate["id"], app_row["id"])
                )

        # Job missing, candidate available
        if not job_id and candidate_id:
            candidate = conn.execute(
                "SELECT job_id FROM candidates WHERE id=?",
                (candidate_id,)
            ).fetchone()

            if candidate and candidate["job_id"]:
                conn.execute(
                    """
                    UPDATE applications
                    SET job_id=?
                    WHERE id=?
                    """,
                    (candidate["job_id"], app_row["id"])
                )


# =========================================================
# PDF TEXT
# =========================================================

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        return text.strip()

    except Exception:
        return ""


# =========================================================
# SKILL NORMALIZATION
# =========================================================

SKILL_ALIASES = {
    "python": [
        "python",
        "python3",
        "python 3"
    ],
    "machine learning": [
        "machine learning",
        "machine-learning",
        "ml"
    ],
    "deep learning": [
        "deep learning",
        "deep-learning",
        "dl"
    ],
    "artificial intelligence": [
        "artificial intelligence",
        "artificial-intelligence",
        "ai"
    ],
    "javascript": [
        "javascript",
        "java script",
        "js"
    ],
    "react": [
        "react",
        "reactjs",
        "react.js"
    ],
    "node": [
        "node",
        "nodejs",
        "node.js"
    ],
    "express": [
        "express",
        "expressjs",
        "express.js"
    ],
    "html": [
        "html",
        "html5"
    ],
    "css": [
        "css",
        "css3"
    ],
    "ms excel": [
        "ms excel",
        "microsoft excel",
        "excel"
    ],
    "ms word": [
        "ms word",
        "microsoft word",
        "word"
    ],
    "power bi": [
        "power bi",
        "powerbi"
    ],
    "data analysis": [
        "data analysis",
        "data analytics"
    ],
    "natural language processing": [
        "natural language processing",
        "nlp"
    ],
    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],
    "c++": [
        "c++",
        "cpp"
    ],
    "c#": [
        "c#",
        "c sharp",
        "csharp"
    ],
    "aws": [
        "aws",
        "amazon web services"
    ],
    "azure": [
        "azure",
        "microsoft azure"
    ],
    "github": [
        "github",
        "git hub"
    ],
    "mysql": [
        "mysql"
    ],
    "postgresql": [
        "postgresql",
        "postgres"
    ],
    "sql": [
        "sql"
    ],
    "pandas": [
        "pandas"
    ],
    "numpy": [
        "numpy"
    ],
    "flask": [
        "flask"
    ],
    "django": [
        "django"
    ],
    "php": [
        "php"
    ],
    "java": [
        "java"
    ],
    "c": [
        "c programming",
        "c language"
    ],
    "git": [
        "git"
    ],
    "powerpoint": [
        "powerpoint",
        "ms powerpoint",
        "microsoft powerpoint",
        "ppt",
        "pptx"
    ],
    "communication": [
        "communication",
        "communication skills"
    ],
    "data science": [
        "data science"
    ],
    "power bi": [
        "power bi",
        "powerbi"
    ],
    "excel": [
        "excel",
        "ms excel",
        "microsoft excel"
    ]
}


def normalize_text(text):
    text = str(text or "").lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_skill(skill):
    skill = normalize_text(skill)

    skill = skill.strip(" ,;|•-")

    return skill


def canonical_skill(skill):
    skill = clean_skill(skill)

    for canonical, aliases in SKILL_ALIASES.items():

        for alias in aliases:
            alias = clean_skill(alias)

            if skill == alias:
                return canonical

    return skill


def parse_required_skills(skills_text):
    if not skills_text:
        return []

    # Support comma, |, semicolon, newline, bullet
    parts = re.split(
        r"[,;|\n•]+",
        str(skills_text)
    )

    result = []

    for part in parts:
        skill = canonical_skill(part)

        if skill and skill not in result:
            result.append(skill)

    return result


def skill_present(skill, resume_text):

    text = str(resume_text or "").lower()

    aliases = SKILL_ALIASES.get(
        skill,
        [skill]
    )

    for alias in aliases:

        alias = str(alias or "").lower().strip()

        if not alias:
            continue

        # Different spellings ko handle karne ke liye
        variants = {
            alias,
            alias.replace("-", " "),
            alias.replace(".", ""),
            alias.replace(".", " "),
        }

        for variant in variants:

            variant = re.sub(
                r"\s+",
                " ",
                variant
            ).strip()

            if not variant:
                continue

            # Special characters jaise C++, C# ko bhi
            # properly match karne ke liye
            pattern = (
                r"(?<![a-z0-9+#])"
                + re.escape(variant)
                + r"(?![a-z0-9+#])"
            )

            if re.search(pattern, text):
                return True

    return False
# =========================================================
# AI MATCHING
# =========================================================

def calculate_match(job_description, required_skills_text, resume_text):

    job_description = str(job_description or "")
    required_skills_text = str(required_skills_text or "")
    resume_text = str(resume_text or "")

    # Required skills ko properly identify karo
    required_skills = parse_required_skills(
        required_skills_text
    )

    matched = []
    missing = []

    # Har required skill ko resume me check karo
    for skill in required_skills:

        if skill_present(skill, resume_text):
            matched.append(skill)
        else:
            missing.append(skill)

    # -----------------------------
    # SKILL MATCH SCORE
    # -----------------------------

    if required_skills:

        skill_score = (
            len(matched) /
            len(required_skills)
        ) * 100

    else:
        skill_score = 0

    # -----------------------------
    # TF-IDF NLP SCORE
    # -----------------------------

    combined_job = (
        job_description
        + " "
        + " ".join(required_skills)
        + " "
        + required_skills_text
    )

    nlp_score = 0

    if (
        combined_job.strip()
        and resume_text.strip()
    ):

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            vectors = vectorizer.fit_transform(
                [
                    combined_job,
                    resume_text
                ]
            )

            nlp_score = (
                cosine_similarity(
                    vectors[0:1],
                    vectors[1:2]
                )[0][0]
                * 100
            )

        except Exception:

            nlp_score = 0

    # -----------------------------
    # FINAL AI MATCH SCORE
    # -----------------------------

    if required_skills:

        final_score = (
            skill_score * 0.70
            +
            nlp_score * 0.30
        )

    else:

        final_score = nlp_score

    # Score ko 0-100 ke andar rakho
    final_score = max(
        0,
        min(
            100,
            round(final_score, 2)
        )
    )

    return (
        final_score,
        matched,
        missing
    )
# =========================================================
# EMAIL EXTRACTION
# =========================================================

def extract_email(text):
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text or ""
    )

    if match:
        return match.group(0).lower()

    return ""


# =========================================================
# SESSION HELPERS
# =========================================================

def current_hr():
    if "hr_db_id" not in session:
        return None

    conn = get_db()

    hr = conn.execute(
        """
        SELECT *
        FROM hr_users
        WHERE id=?
        """,
        (session["hr_db_id"],)
    ).fetchone()

    conn.close()

    return hr


def current_candidate():
    if "candidate_id" not in session:
        return None

    conn = get_db()

    candidate = conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE id=?
        """,
        (session["candidate_id"],)
    ).fetchone()

    conn.close()

    return candidate


# =========================================================
# WEBSITE HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# HR LOGIN
# =========================================================

@app.route("/hr/login", methods=["GET", "POST"])
def hr_login():

    if request.method == "POST":

        hr_id = request.form.get("hr_id", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()

        hr = conn.execute(
            """
            SELECT *
            FROM hr_users
            WHERE hr_id=?
            AND password=?
            AND status='Active'
            """,
            (hr_id, password)
        ).fetchone()

        conn.close()

        if not hr:
            flash("Invalid HR ID or password.", "danger")
            return redirect(url_for("hr_login"))

        session.clear()

        session["hr_db_id"] = hr["id"]
        session["hr_id"] = hr["hr_id"]
        session["hr_name"] = hr["name"]

        return redirect(url_for("dashboard"))

    return render_template("hr_login.html")


# =========================================================
# HR LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("hr_login"))


# =========================================================
# HR DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    hr_db_id = hr["id"]

    conn = get_db()

    # HR's jobs only
    jobs = conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE hr_id=?
        ORDER BY id DESC
        """,
        (hr_db_id,)
    ).fetchall()

    # Total applications/resumes
    total_resumes = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN jobs j ON j.id=a.job_id
        WHERE j.hr_id=?
        """,
        (hr_db_id,)
    ).fetchone()[0]

    # Selected
    selected_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN jobs j ON j.id=a.job_id
        WHERE j.hr_id=?
        AND LOWER(COALESCE(a.status,'')) IN
        ('selected','shortlisted')
        """,
        (hr_db_id,)
    ).fetchone()[0]

    # Under Review
    review_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN jobs j ON j.id=a.job_id
        WHERE j.hr_id=?
        AND LOWER(COALESCE(a.status,'')) IN
        ('under review','under_review','review')
        """,
        (hr_db_id,)
    ).fetchone()[0]

    # Not matching / rejected
    not_matching_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN jobs j ON j.id=a.job_id
        WHERE j.hr_id=?
        AND LOWER(COALESCE(a.status,'')) IN
        ('not matching','not_matched','rejected','not selected')
        """,
        (hr_db_id,)
    ).fetchone()[0]

    # Existing template compatibility
    application_count = total_resumes

    conn.close()

    return render_template(
        "dashboard.html",
        jobs=jobs,

        # Dashboard cards
        total_candidates=total_resumes,
        shortlisted=selected_count,
        review=review_count,
        not_matching=not_matching_count,

        # Compatibility
        total_resumes=total_resumes,
        shortlisted_count=selected_count,
        selected_count=selected_count,
        review_count=review_count,
        under_review_count=review_count,
        not_matching_count=not_matching_count,
        application_count=application_count,

        # HR information
        hr_name=hr["name"],
        hr_id=hr["hr_id"]
    )


# =========================================================
# CREATE JOB
# =========================================================

@app.route("/create-job", methods=["GET", "POST"])
def create_job():

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    if request.method == "POST":

        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        skills = request.form.get("skills", "").strip()

        if not title:
            flash("Job title is required.", "danger")
            return redirect(url_for("create_job"))

        conn = get_db()

        conn.execute(
            """
            INSERT INTO jobs
            (title,description,skills,hr_id)
            VALUES (?,?,?,?)
            """,
            (
                title,
                description,
                skills,
                hr["id"]
            )
        )

        conn.commit()
        conn.close()

        flash("Job created successfully.", "success")

        return redirect(url_for("dashboard"))

    return render_template("create_job.html")


# =========================================================
# SCREEN RESUMES
# =========================================================

@app.route("/screen/<int:job_id>", methods=["GET", "POST"])
def screen_resumes(job_id):

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    conn = get_db()

    job = conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE id=?
        AND hr_id=?
        """,
        (job_id, hr["id"])
    ).fetchone()

    if not job:
        conn.close()
        flash("Job not found or access denied.", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        files = request.files.getlist("resumes")

        if not files:
            single = request.files.get("resume")

            if single:
                files = [single]

        processed = 0
        skipped = 0

        for file in files:

            if not file or not file.filename:
                continue

            original_filename = secure_filename(
                file.filename
            )

            if not original_filename.lower().endswith(".pdf"):
                skipped += 1
                continue

            unique_filename = (
                str(uuid.uuid4()) +
                "_" +
                original_filename
            )

            file_path = os.path.join(
                UPLOAD_FOLDER,
                unique_filename
            )

            file.save(file_path)

            resume_text = extract_text_from_pdf(
                file_path
            )

            candidate_name = (
                request.form.get("candidate_name")
                or "Candidate"
            ).strip()

            email = extract_email(resume_text)

            if not email:
                email = (
                    "resume_" +
                    uuid.uuid4().hex +
                    "@local.invalid"
                )

            # Existing candidate by email
            candidate = find_candidate_by_email(
                conn,
                email
            )

            if candidate:

                candidate_id = candidate["id"]

                conn.execute(
                    """
                    UPDATE candidates
                    SET
                        name=?,
                        email=?,
                        filename=?,
                        resume=?,
                        resume_text=?
                    WHERE id=?
                    """,
                    (
                        candidate_name,
                        email,
                        unique_filename,
                        unique_filename,
                        resume_text,
                        candidate_id
                    )
                )

            else:

                cursor = conn.execute(
                    """
                    INSERT INTO candidates
                    (
                        job_id,
                        filename,
                        name,
                        email,
                        resume,
                        resume_text,
                        status
                    )
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        unique_filename,
                        candidate_name,
                        email,
                        unique_filename,
                        resume_text,
                        "Under Review"
                    )
                )

                candidate_id = cursor.lastrowid

            # Find existing candidate application
            existing = conn.execute(
                """
                SELECT *
                FROM applications
                WHERE job_id=?
                AND candidate_id=?
                LIMIT 1
                """,
                (
                    job_id,
                    candidate_id
                )
            ).fetchone()

            if existing:
                skipped += 1
                continue

            # AI score
            score, matched, missing = calculate_match(
                job["description"],
                job["skills"],
                resume_text
            )

            matched_text = ", ".join(matched)
            missing_text = ", ".join(missing)

            # Find user if candidate has one
            user_id = None

            user = conn.execute(
                """
                SELECT id
                FROM users
                WHERE LOWER(COALESCE(email,''))=?
                LIMIT 1
                """,
                (email.lower(),)
            ).fetchone()

            if user:
                user_id = user["id"]

            # IMPORTANT:
            # legacy DB may have UNIQUE(user_id,job_id)
            # so check it before insert
            if user_id:

                existing_user_app = conn.execute(
                    """
                    SELECT *
                    FROM applications
                    WHERE job_id=?
                    AND user_id=?
                    LIMIT 1
                    """,
                    (
                        job_id,
                        user_id
                    )
                ).fetchone()

                if existing_user_app:
                    skipped += 1
                    continue

            # Insert application
            conn.execute(
                """
                INSERT INTO applications
                (
                    user_id,
                    job_id,
                    candidate_id,
                    filename,
                    resume,
                    ai_score,
                    matched_skills,
                    missing_skills,
                    status,
                    applied_at
                )
                VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                """,
                (
                    user_id,
                    job_id,
                    candidate_id,
                    unique_filename,
                    unique_filename,
                    score,
                    matched_text,
                    missing_text,
                    "Under Review"
                )
            )

            # Candidate table = latest snapshot only
            conn.execute(
                """
                UPDATE candidates
                SET
                    job_id=?,
                    match_percentage=?,
                    matched_skills=?,
                    missing_skills=?,
                    status=?
                WHERE id=?
                """,
                (
                    job_id,
                    score,
                    matched_text,
                    missing_text,
                    "Under Review",
                    candidate_id
                )
            )

            processed += 1

        conn.commit()
        conn.close()

        flash(
            f"{processed} resume(s) processed successfully. "
            f"{skipped} duplicate/invalid resume(s) skipped.",
            "success"
        )

        return redirect(
            url_for(
                "results",
                job_id=job_id
            )
        )

    conn.close()

    return render_template(
        "screen.html",
        job=job
    )


# =========================================================
# HR RESULTS
# =========================================================

@app.route("/results/<int:job_id>")
def results(job_id):

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    conn = get_db()

    job = conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE id=?
        AND hr_id=?
        """,
        (
            job_id,
            hr["id"]
        )
    ).fetchone()

    if not job:
        conn.close()
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    applications = conn.execute(
        """
        SELECT
            a.id,
            a.candidate_id,
            a.user_id,
            a.job_id,
            a.filename,
            a.resume,
            a.ai_score,
            a.matched_skills,
            a.missing_skills,
            a.status,
            a.applied_at,
            c.name AS candidate_name,
            c.email AS candidate_email,
            c.mobile AS candidate_mobile,
            j.title AS job_title
        FROM applications a
        JOIN jobs j
            ON j.id=a.job_id
        LEFT JOIN candidates c
            ON c.id=a.candidate_id
        WHERE a.job_id=?
        AND j.hr_id=?
        ORDER BY a.id DESC
        """,
        (
            job_id,
            hr["id"]
        )
    ).fetchall()

    conn.close()

    return render_template(
        "results.html",
        job=job,
        candidates=applications,
        applications=applications,
        hr_name=hr["name"],
        hr_id=hr["hr_id"]
    )


# =========================================================
# HR APPLICATIONS
# =========================================================

@app.route("/hr/applications/<int:job_id>")
def hr_applications(job_id):

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    conn = get_db()

    job = conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE id=?
        AND hr_id=?
        """,
        (
            job_id,
            hr["id"]
        )
    ).fetchone()

    if not job:
        conn.close()
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    applications = conn.execute(
        """
        SELECT
            a.*,
            c.name AS candidate_name,
            c.email AS candidate_email,
            c.mobile AS candidate_mobile,
            j.title AS job_title
        FROM applications a
        JOIN jobs j
            ON j.id=a.job_id
        LEFT JOIN candidates c
            ON c.id=a.candidate_id
        WHERE a.job_id=?
        AND j.hr_id=?
        ORDER BY a.id DESC
        """,
        (
            job_id,
            hr["id"]
        )
    ).fetchall()

    conn.close()

    return render_template(
        "applications.html",
        job=job,
        applications=applications,
        hr_name=hr["name"],
        hr_id=hr["hr_id"]
    )


# =========================================================
# HR SELECT / REJECT
# =========================================================

@app.route(
    "/hr/application/<int:application_id>/<action>",
    methods=["GET", "POST"]
)
def hr_application_action(
    application_id,
    action
):

    hr = current_hr()

    if not hr:
        return redirect(url_for("hr_login"))

    action = action.lower().strip()

    if action in ["select", "selected", "shortlist"]:
        new_status = "Selected"

    elif action in ["reject", "rejected"]:
        new_status = "Not Selected"

    elif action in ["review", "under-review"]:
        new_status = "Under Review"

    else:
        flash("Invalid action.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db()

    application = conn.execute(
        """
        SELECT
            a.*,
            j.hr_id,
            j.title
        FROM applications a
        JOIN jobs j
            ON j.id=a.job_id
        WHERE a.id=?
        AND j.hr_id=?
        """,
        (
            application_id,
            hr["id"]
        )
    ).fetchone()

    if not application:
        conn.close()
        flash("Application not found.", "danger")
        return redirect(url_for("dashboard"))

    conn.execute(
        """
        UPDATE applications
        SET status=?
        WHERE id=?
        """,
        (
            new_status,
            application_id
        )
    )

    # Keep candidate legacy snapshot synchronized,
    # but application remains the source of truth.
    if application["candidate_id"]:
        conn.execute(
            """
            UPDATE candidates
            SET status=?
            WHERE id=?
            """,
            (
                new_status,
                application["candidate_id"]
            )
        )

    conn.commit()
    conn.close()

    flash(
        f"Application status changed to {new_status}.",
        "success"
    )

    return redirect(
        request.referrer
        or url_for(
            "hr_applications",
            job_id=application["job_id"]
        )
    )


# =========================================================
# CANDIDATE REGISTER
# =========================================================

@app.route(
    "/candidate/register",
    methods=["GET", "POST"]
)
def candidate_register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not name or not mobile or not password:
            flash(
                "Name, mobile and password are required.",
                "danger"
            )
            return redirect(
                url_for("candidate_register")
            )

        try:

            user_id = get_or_create_user(
                name,
                email,
                mobile,
                password
            )

            conn = get_db()

            candidate = find_candidate_by_user(
                conn,
                user_id
            )

            if not candidate:

                cursor = conn.execute(
                    """
                    INSERT INTO candidates
                    (
                        name,
                        email,
                        mobile,
                        password,
                        status
                    )
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        name,
                        email,
                        mobile,
                        password,
                        "Under Review"
                    )
                )

                candidate_id = cursor.lastrowid

            else:

                candidate_id = candidate["id"]

                conn.execute(
                    """
                    UPDATE candidates
                    SET
                        name=?,
                        email=?,
                        mobile=?,
                        password=?
                    WHERE id=?
                    """,
                    (
                        name,
                        email,
                        mobile,
                        password,
                        candidate_id
                    )
                )

            conn.commit()
            conn.close()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(
                url_for("candidate_login")
            )

        except sqlite3.IntegrityError as e:

            flash(
                "Registration failed. Mobile/email may already exist.",
                "danger"
            )

            return redirect(
                url_for("candidate_register")
            )

    return render_template(
        "candidate_register.html"
    )


# =========================================================
# CANDIDATE LOGIN
# =========================================================

@app.route(
    "/candidate/login",
    methods=["GET", "POST"]
)
def candidate_login():

    if request.method == "POST":

        mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE mobile=?
            AND password=?
            LIMIT 1
            """,
            (
                mobile,
                password
            )
        ).fetchone()

        if not user:

            # Email login fallback
            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE LOWER(email)=?
                AND password=?
                LIMIT 1
                """,
                (
                    mobile.lower(),
                    password
                )
            ).fetchone()

        if not user:
            conn.close()

            flash(
                "Invalid mobile/email or password.",
                "danger"
            )

            return redirect(
                url_for("candidate_login")
            )

        candidate = find_candidate_by_user(
            conn,
            user["id"]
        )

        if not candidate:

            cursor = conn.execute(
                """
                INSERT INTO candidates
                (
                    name,
                    email,
                    mobile,
                    password,
                    status
                )
                VALUES (?,?,?,?,?)
                """,
                (
                    user["name"],
                    user["email"],
                    user["mobile"],
                    user["password"],
                    "Under Review"
                )
            )

            candidate_id = cursor.lastrowid

        else:
            candidate_id = candidate["id"]

        conn.commit()
        conn.close()

        session.clear()

        session["candidate_id"] = candidate_id
        session["candidate_user_id"] = user["id"]
        session["candidate_name"] = (
            user["name"] or "Candidate"
        )

        return redirect(
            url_for("candidate_dashboard")
        )

    return render_template(
        "candidate_login.html"
    )


# =========================================================
# CANDIDATE LOGOUT
# =========================================================

@app.route("/candidate/logout")
def candidate_logout():

    session.clear()

    return redirect(
        url_for("candidate_login")
    )


# =========================================================
# CANDIDATE DASHBOARD
# =========================================================

@app.route("/candidate/dashboard")
def candidate_dashboard():

    if "candidate_id" not in session:
        return redirect(
            url_for("candidate_login")
        )

    candidate_id = session["candidate_id"]
    user_id = session.get("candidate_user_id")

    conn = get_db()

    candidate = conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE id=?
        """,
        (candidate_id,)
    ).fetchone()

    # Active HR jobs
    jobs = conn.execute(
        """
        SELECT
            j.*,
            h.name AS hr_name,
            h.hr_id AS hr_code
        FROM jobs j
        JOIN hr_users h
            ON h.id=j.hr_id
        WHERE h.status='Active'
        ORDER BY j.id DESC
        """
    ).fetchall()

    # Candidate history
    history = conn.execute(
        """
        SELECT
            a.id,
            a.job_id,
            a.status,
            a.applied_at,
            j.title AS job_title,
            h.name AS hr_name
        FROM applications a
        JOIN jobs j
            ON j.id=a.job_id
        LEFT JOIN hr_users h
            ON h.id=j.hr_id
        WHERE
            a.candidate_id=?
            OR
            a.user_id=?
        ORDER BY a.id DESC
        """,
        (
            candidate_id,
            user_id
        )
    ).fetchall()

    conn.close()

    candidate_name = (
        candidate["name"]
        if candidate and candidate["name"]
        else session.get(
            "candidate_name",
            "Candidate"
        )
    )

    return render_template(
        "candidate_dashboard.html",
        candidate=candidate,
        candidate_name=candidate_name,
        jobs=jobs,
        history=history
    )


# =========================================================
# CANDIDATE APPLY
# =========================================================

@app.route(
    "/candidate/apply/<int:job_id>",
    methods=["GET", "POST"]
)
def candidate_apply(job_id):

    if "candidate_id" not in session:
        return redirect(
            url_for("candidate_login")
        )

    candidate_id = session["candidate_id"]
    user_id = session.get("candidate_user_id")

    conn = get_db()

    job = conn.execute(
        """
        SELECT
            j.*,
            h.name AS hr_name,
            h.hr_id AS hr_code
        FROM jobs j
        JOIN hr_users h
            ON h.id=j.hr_id
        WHERE j.id=?
        AND h.status='Active'
        """,
        (job_id,)
    ).fetchone()

    if not job:
        conn.close()

        flash(
            "Job not found or no longer available.",
            "danger"
        )

        return redirect(
            url_for("candidate_dashboard")
        )

    # IMPORTANT:
    # Check candidate_id + job_id
    existing_candidate = conn.execute(
        """
        SELECT *
        FROM applications
        WHERE candidate_id=?
        AND job_id=?
        LIMIT 1
        """,
        (
            candidate_id,
            job_id
        )
    ).fetchone()

    if existing_candidate:
        conn.close()

        flash(
            "You have already applied for this job.",
            "warning"
        )

        return redirect(
            url_for("candidate_applications")
        )

    # IMPORTANT:
    # Check user_id + job_id also because legacy DB
    # can contain UNIQUE(user_id, job_id)
    if user_id:

        existing_user = conn.execute(
            """
            SELECT *
            FROM applications
            WHERE user_id=?
            AND job_id=?
            LIMIT 1
            """,
            (
                user_id,
                job_id
            )
        ).fetchone()

        if existing_user:
            conn.close()

            flash(
                "You have already applied for this job.",
                "warning"
            )

            return redirect(
                url_for("candidate_applications")
            )

    if request.method == "POST":

        file = request.files.get("resume")

        if not file or not file.filename:

            conn.close()

            flash(
                "Please upload your resume PDF.",
                "danger"
            )

            return redirect(
                url_for(
                    "candidate_apply",
                    job_id=job_id
                )
            )

        original_filename = secure_filename(
            file.filename
        )

        if not original_filename.lower().endswith(".pdf"):

            conn.close()

            flash(
                "Only PDF resumes are allowed.",
                "danger"
            )

            return redirect(
                url_for(
                    "candidate_apply",
                    job_id=job_id
                )
            )

        unique_filename = (
            str(uuid.uuid4()) +
            "_" +
            original_filename
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )

        file.save(file_path)

        resume_text = extract_text_from_pdf(
            file_path
        )

        candidate = conn.execute(
            """
            SELECT *
            FROM candidates
            WHERE id=?
            """,
            (candidate_id,)
        ).fetchone()

        candidate_name = (
            candidate["name"]
            if candidate
            else session.get(
                "candidate_name",
                "Candidate"
            )
        )

        email = (
            candidate["email"]
            if candidate
            else ""
        )

        # AI score
        score, matched, missing = calculate_match(
            job["description"],
            job["skills"],
            resume_text
        )

        matched_text = ", ".join(matched)
        missing_text = ", ".join(missing)

        # Insert application
        try:

            conn.execute(
                """
                INSERT INTO applications
                (
                    user_id,
                    job_id,
                    candidate_id,
                    filename,
                    resume,
                    ai_score,
                    matched_skills,
                    missing_skills,
                    status,
                    applied_at
                )
                VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                """,
                (
                    user_id,
                    job_id,
                    candidate_id,
                    unique_filename,
                    unique_filename,
                    score,
                    matched_text,
                    missing_text,
                    "Under Review"
                )
            )

        except sqlite3.IntegrityError:

            conn.close()

            # Remove uploaded duplicate file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

            flash(
                "You have already applied for this job.",
                "warning"
            )

            return redirect(
                url_for("candidate_applications")
            )

        # Candidate legacy snapshot only
        conn.execute(
            """
            UPDATE candidates
            SET
                job_id=?,
                filename=?,
                resume=?,
                resume_text=?,
                match_percentage=?,
                matched_skills=?,
                missing_skills=?,
                status=?
            WHERE id=?
            """,
            (
                job_id,
                unique_filename,
                unique_filename,
                resume_text,
                score,
                matched_text,
                missing_text,
                "Under Review",
                candidate_id
            )
        )

        conn.commit()
        conn.close()

        flash(
            "Application submitted successfully.",
            "success"
        )

        return redirect(
            url_for("candidate_applications")
        )

    conn.close()

    return render_template(
        "candidate_apply.html",
        job=job
    )


# =========================================================
# CANDIDATE APPLICATION HISTORY
# =========================================================

@app.route("/candidate/applications")
def candidate_applications():

    if "candidate_id" not in session:
        return redirect(
            url_for("candidate_login")
        )

    candidate_id = session["candidate_id"]
    user_id = session.get("candidate_user_id")

    conn = get_db()

    # IMPORTANT:
    # Do NOT SELECT a.* here.
    # This prevents AI score/matched/missing skills
    # from being exposed to candidate.
    applications = conn.execute(
        """
        SELECT
            a.id,
            a.job_id,
            a.status,
            a.applied_at,
            j.title AS job_title,
            j.description AS job_description,
            h.name AS hr_name,
            h.hr_id AS hr_code
        FROM applications a
        JOIN jobs j
            ON j.id=a.job_id
        LEFT JOIN hr_users h
            ON h.id=j.hr_id
        WHERE
            a.candidate_id=?
            OR
            a.user_id=?
        ORDER BY a.id DESC
        """,
        (
            candidate_id,
            user_id
        )
    ).fetchall()

    candidate = conn.execute(
        """
        SELECT *
        FROM candidates
        WHERE id=?
        """,
        (candidate_id,)
    ).fetchone()

    conn.close()

    return render_template(
        "candidate_applications.html",
        applications=applications,
        candidate=candidate,
        candidate_name=(
            candidate["name"]
            if candidate and candidate["name"]
            else session.get(
                "candidate_name",
                "Candidate"
            )
        )
    )


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.route(
    "/admin",
    methods=["GET", "POST"]
)
@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if (
            username == "admin"
            and
            password == "admin123"
        ):

            session.clear()

            session["admin_logged_in"] = True
            session["admin_name"] = "System Administrator"

            return redirect(
                url_for("admin_dashboard")
            )

        flash(
            "Invalid admin username or password.",
            "danger"
        )

    return render_template(
        "admin_login.html"
    )


# =========================================================
# ADMIN LOGOUT
# =========================================================

@app.route("/admin/logout")
def admin_logout():

    session.clear()

    return redirect(
        url_for("admin_login")
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin/dashboard")
def admin_dashboard():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db()

    # ==============================
    # BASIC COUNTS
    # ==============================

    hr_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM hr_users
        """
    ).fetchone()[0]

    active_hr_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM hr_users
        WHERE status='Active'
        """
    ).fetchone()[0]

    inactive_hr_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM hr_users
        WHERE status!='Active'
        """
    ).fetchone()[0]

    job_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM jobs
        """
    ).fetchone()[0]

    application_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications
        """
    ).fetchone()[0]

    candidate_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        """
    ).fetchone()[0]

    selected_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications
        WHERE status='Selected'
        """
    ).fetchone()[0]

    not_selected_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications
        WHERE status='Not Selected'
        """
    ).fetchone()[0]

    review_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM applications
        WHERE status='Under Review'
        """
    ).fetchone()[0]

    # ==============================
    # HR-WISE PERFORMANCE
    # ==============================

    hr_performance = conn.execute(
        """
        SELECT
            h.id,
            h.hr_id,
            h.name,
            h.status,

            COUNT(DISTINCT j.id) AS jobs,

            COUNT(DISTINCT a.id) AS applications,

            COUNT(
                DISTINCT CASE
                    WHEN a.status='Selected'
                    THEN a.id
                END
            ) AS selected,

            COUNT(
                DISTINCT CASE
                    WHEN a.status='Not Selected'
                    THEN a.id
                END
            ) AS not_selected,

            COUNT(
                DISTINCT CASE
                    WHEN a.status='Under Review'
                    THEN a.id
                END
            ) AS under_review

        FROM hr_users h

        LEFT JOIN jobs j
            ON j.hr_id = h.id

        LEFT JOIN applications a
            ON a.job_id = j.id

        GROUP BY
            h.id,
            h.hr_id,
            h.name,
            h.status

        ORDER BY h.id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
    "admin_dashboard.html",

    admin_name=session.get(
        "admin_name",
        "System Administrator"
    ),

    # Original names
    hr_count=hr_count,
    active_hr_count=active_hr_count,
    inactive_hr_count=inactive_hr_count,

    job_count=job_count,
    application_count=application_count,
    candidate_count=candidate_count,

    selected_count=selected_count,
    not_selected_count=not_selected_count,
    review_count=review_count,

    hr_performance=hr_performance,

    # Template compatibility names
    total_hr=hr_count,
    active_hr=active_hr_count,
    inactive_hr=inactive_hr_count,

    total_jobs=job_count,
    total_candidates=candidate_count,
    total_applications=application_count,

    selected=selected_count,
    not_selected=not_selected_count,
    under_review=review_count,

    # HR table compatibility
    hr_data=hr_performance
)



# =========================================================
# ADMIN HR LIST
# =========================================================

@app.route("/admin/hr")
def admin_hr():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db()

    hrs = conn.execute(
        """
        SELECT *
        FROM hr_users
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "admin_hr.html",
        hrs=hrs,
        admin_name=session.get(
            "admin_name",
            "System Administrator"
        )
    )


# =========================================================
# ADMIN CREATE HR
# =========================================================

@app.route(
    "/admin/hr/create",
    methods=["GET", "POST"]
)
def admin_create_hr():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not name or not password:
            flash(
                "HR name and password are required.",
                "danger"
            )

            return redirect(
                url_for("admin_create_hr")
            )

        conn = get_db()

        # Find maximum numeric HR suffix
        rows = conn.execute(
            """
            SELECT hr_id
            FROM hr_users
            WHERE hr_id IS NOT NULL
            """
        ).fetchall()

        max_number = 0

        for row in rows:

            hr_id = str(
                row["hr_id"] or ""
            ).upper()

            match = re.match(
                r"HR(\d+)$",
                hr_id
            )

            if match:
                number = int(
                    match.group(1)
                )

                if number > max_number:
                    max_number = number

        new_hr_id = (
            "HR" +
            str(max_number + 1).zfill(3)
        )

        conn.execute(
            """
            INSERT INTO hr_users
            (
                hr_id,
                name,
                email,
                password,
                status
            )
            VALUES (?,?,?,?,?)
            """,
            (
                new_hr_id,
                name,
                email,
                password,
                "Active"
            )
        )

        conn.commit()
        conn.close()

        flash(
            f"HR created successfully. HR ID: {new_hr_id}",
            "success"
        )

        return redirect(
            url_for("admin_hr")
        )

    return render_template(
        "admin_create_hr.html"
    )


# =========================================================
# ADMIN TOGGLE HR
# =========================================================

@app.route(
    "/admin/hr/toggle/<int:hr_db_id>"
)
def admin_toggle_hr(hr_db_id):

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db()

    hr = conn.execute(
        """
        SELECT *
        FROM hr_users
        WHERE id=?
        """,
        (hr_db_id,)
    ).fetchone()

    if hr:

        new_status = (
            "Inactive"
            if hr["status"] == "Active"
            else "Active"
        )

        conn.execute(
            """
            UPDATE hr_users
            SET status=?
            WHERE id=?
            """,
            (
                new_status,
                hr_db_id
            )
        )

        conn.commit()

    conn.close()

    return redirect(
        url_for("admin_hr")
    )


# =========================================================
# ADMIN DELETE / DEACTIVATE HR
# =========================================================

@app.route(
    "/admin/hr/delete/<int:hr_db_id>"
)
def admin_delete_hr(hr_db_id):

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db()

    # IMPORTANT:
    # Do not delete jobs/applications.
    # Keep history safe.
    conn.execute(
        """
        UPDATE jobs
        SET hr_id=NULL
        WHERE hr_id=?
        """,
        (hr_db_id,)
    )

    conn.execute(
        """
        UPDATE hr_users
        SET status='Inactive'
        WHERE id=?
        """,
        (hr_db_id,)
    )

    conn.commit()
    conn.close()

    flash(
        "HR deactivated. Existing application history was preserved.",
        "success"
    )

    return redirect(
        url_for("admin_hr")
    )


# =========================================================
# ADMIN JOBS
# =========================================================

@app.route("/admin/jobs")
def admin_jobs():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db()

    jobs = conn.execute(
        """
        SELECT
            j.*,
            h.name AS hr_name,
            h.hr_id AS hr_code
        FROM jobs j
        LEFT JOIN hr_users h
            ON h.id=j.hr_id
        ORDER BY j.id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "admin_jobs.html",
        jobs=jobs,
        admin_name=session.get(
            "admin_name",
            "System Administrator"
        )
    )


# =========================================================
# START
# =========================================================

# Initialize database when app starts
# This is required for Render / Gunicorn
init_db()


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )