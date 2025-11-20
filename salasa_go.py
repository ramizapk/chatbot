import streamlit as st
import uuid

# Initialize session state at the VERY BEGINNING
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_language" not in st.session_state:
    st.session_state.current_language = 'english'
if "admin_dashboard_expanded" not in st.session_state:
    st.session_state.admin_dashboard_expanded = False
if "show_advanced_analytics" not in st.session_state:
    st.session_state.show_advanced_analytics = False
if 'file_to_download' not in st.session_state:
    st.session_state.file_to_download = None
if 'file_to_delete' not in st.session_state:
    st.session_state.file_to_delete = None
if 'show_delete_confirm' not in st.session_state:
    st.session_state.show_delete_confirm = False

# Now continue with the rest of your imports and code
import re
import os
import json
import uuid
from datetime import datetime, timedelta
from collections import Counter
from langdetect import detect
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pymongo import MongoClient
from gridfs import GridFS
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


def send_conversation_email(to_email, conversation_text):
    sender_email = "universiy.b@outlook.com"
    app_password = "79JC3-68KHN-TK7TE-XN5X8-XVMM9"  # move to st.secrets in production

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = "Your Conversation History -"
    msg.attach(MIMEText(conversation_text, "plain"))

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        return True
    except Exception as e:
        return str(e)

# === Configuration ===
import os
from pymongo import MongoClient
from gridfs import GridFS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama

INDEX_DIR = "faiss_index"
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminpass"
MONGO_URI = "mongodb://localhost:27017"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["university_chatbot"]
fs = GridFS(db)

files_collection = db["files"]
convo_collection = db["conversations"]

# Check for missing files and re-upload if needed
missing_files = db.file_metadata.find({"file_id": None})
for f in missing_files:
    filename = f["filename"]
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        # Use context manager to safely open file
        with open(file_path, "rb") as file_data:
            new_file_id = fs.put(file_data, filename=filename)
        # Update metadata with new GridFS ID
        db.file_metadata.update_one(
            {"_id": f["_id"]},
            {"$set": {"file_id": new_file_id}}
        )
        print(f"Uploaded missing file '{filename}' to GridFS with ID {new_file_id}")
    else:
        print(f"Missing file on disk: {filename}")
    
# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Initialize LLM
llm = Ollama(model="llama3", temperature=0, num_predict=2048)
    

# === Helper Functions ===
# New version with proper sync
# Enhanced file upload function that ensures proper sync
def rebuild_vector_store_without_file(filename_to_remove):
    """Rebuild the vector store excluding a specific file"""
    try:
        if not faiss_index_exists():
            return "No vector store found"
        
        # Load current vector store
        vectorstore = load_faiss()
        
        # Get all documents except those from the file to remove
        new_docs = []
        for doc_id in vectorstore.docstore._dict:
            doc = vectorstore.docstore._dict[doc_id]
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                if doc.metadata['source'] != filename_to_remove:
                    new_docs.append(doc)
        
        # Recreate vector store
        if new_docs:
            new_store = FAISS.from_documents(new_docs, embedding_model)
            new_store.save_local(INDEX_DIR)
            return f"Vector store rebuilt without {filename_to_remove}"
        else:
            # If no documents left, delete the index
            if os.path.exists(INDEX_DIR):
                import shutil
                shutil.rmtree(INDEX_DIR)
            return "Vector store deleted as no documents remain"
            
    except Exception as e:
        return f"Error rebuilding vector store: {str(e)}"

def check_and_fix_sync_issues():
    """Comprehensive sync check and fix function"""
    try:
        # Find files with missing file_id
        broken_files = list(files_collection.find({"file_id": None}))
        
        if not broken_files:
            return {"status": "success", "message": "No sync issues found", "fixed_count": 0, "total_broken": 0}
        
        fixed_count = 0
        for file_record in broken_files:
            filename = file_record["filename"]
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            if os.path.exists(file_path):
                # Upload to GridFS
                with open(file_path, "rb") as f:
                    file_id = fs.put(f, filename=filename)
                
                # Update database record
                files_collection.update_one(
                    {"_id": file_record["_id"]},
                    {"$set": {"file_id": file_id}}
                )
                fixed_count += 1
                st.success(f"Fixed {filename}")
            else:
                st.warning(f"File not found on disk: {filename}")
        
        return {
            "status": "success",
            "fixed_count": fixed_count,
            "total_broken": len(broken_files)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
def save_uploaded_file_with_sync(uploaded_file):
    """Save uploaded file ensuring both GridFS and database record are created"""
    try:
        # First save to local directory as backup
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Save to GridFS
        file_id = fs.put(uploaded_file.getvalue(), filename=uploaded_file.name)
        
        # Create database record
        file_doc = {
            "filename": uploaded_file.name,
            "file_id": file_id,
            "uploaded_at": datetime.now(),
            "status": "complete",
            "file_size": len(uploaded_file.getvalue()),
            "local_backup_path": file_path
        }
        
        db_id = files_collection.insert_one(file_doc).inserted_id
        
        return {
            "success": True,
            "db_id": db_id,
            "file_id": file_id,
            "filename": uploaded_file.name
        }
        
    except Exception as e:
        # Cleanup if anything failed
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
            
        try:
            if 'file_id' in locals():
                fs.delete(file_id)
        except:
            pass
            
        return {
            "success": False,
            "error": str(e)
        }
def get_file_for_download(filename):
    """Get file data with proper error handling"""
    try:
        file_record = files_collection.find_one({"filename": filename})
        
        if not file_record:
            return {"success": False, "message": "File not found in database"}
        
        # Try GridFS first
        if file_record.get("file_id"):
            try:
                gridfs_file = fs.get(file_record["file_id"])
                file_data = gridfs_file.read()
                return {"success": True, "data": file_data}
            except Exception as gridfs_error:
                st.warning(f"GridFS error: {gridfs_error}")
                # Fall through to local backup
        
        # Try local backup
        local_path = file_record.get("local_backup_path", os.path.join(UPLOAD_DIR, filename))
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                file_data = f.read()
            return {"success": True, "data": file_data}
        
        return {"success": False, "message": "File not found in any storage location"}
        
    except Exception as e:
        return {"success": False, "message": f"Error retrieving file: {str(e)}"}



def load_content_with_ocr(uploaded_file):
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.lower()

    if file_ext.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image, lang="ara+eng")

    elif file_ext.endswith(".pdf"):
        images = convert_from_bytes(file_bytes)
        return "\n".join(pytesseract.image_to_string(img, lang="ara+eng") for img in images)

    elif file_ext.endswith(".txt"):
        return file_bytes.decode("utf-8")

    elif file_ext.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)

    return ""

def make_documents(text, source):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

def load_faiss_or_create(docs):
    if os.path.exists(INDEX_DIR):
        store = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        store.add_documents(docs)
        store.save_local(INDEX_DIR)
    else:
        store = FAISS.from_documents(docs, embedding_model)
        os.makedirs(INDEX_DIR, exist_ok=True)
        store.save_local(INDEX_DIR)
    return store

def load_faiss():
    return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)

def faiss_index_exists() -> bool:
    if not (os.path.isdir(INDEX_DIR) and any(os.scandir(INDEX_DIR))):
        return False
    try:
        _ = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        return True
    except Exception:
        return False
    
def get_vectorstore_or_none():
    try:
        return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    except Exception:
        return None

def save_conversation(user_input, bot_response):
    convo_collection.insert_one({
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now(),
        "user_message": user_input,
        "bot_response": bot_response
    })

def get_recent_conversations(limit=10):
    return convo_collection.find().sort("timestamp", -1).limit(10)

def generate_conversation_report():
    """Generate conversation analytics report with proper error handling"""
    try:
        # Get all conversations grouped by session
        sessions = list(convo_collection.aggregate([
            {"$group": {
                "_id": "$session_id",
                "start_time": {"$min": "$timestamp"},
                "end_time": {"$max": "$timestamp"},
                "questions": {"$push": "$user_message"}
            }}
        ]))
        
        session_list = sessions
        total_sessions = len(session_list)
        total_duration = 0
        total_questions = 0
        question_counter = Counter()

        for s in session_list:
            # Calculate session duration in minutes
            if "start_time" in s and "end_time" in s:
                duration = (s["end_time"] - s["start_time"]).total_seconds() / 60
                total_duration += duration
            
            # Count questions
            if "questions" in s:
                num_qs = len(s["questions"])
                total_questions += num_qs
                question_counter.update(s["questions"])

        # Calculate averages (avoid division by zero)
        avg_duration = total_duration / total_sessions if total_sessions else 0
        avg_questions = total_questions / total_sessions if total_sessions else 0
        
        # Get most common questions
        most_common_questions = question_counter.most_common(5)
        
        # Count total files
        total_files = files_collection.count_documents({})

        return {
            "total_sessions": total_sessions,
            "avg_session_duration_min": round(avg_duration, 2),
            "avg_questions_per_session": round(avg_questions, 2),
            "total_uploaded_files": total_files,
            "most_common_questions": most_common_questions
        }
        
    except Exception as e:
        # Return default values if there's an error
        st.error(f"Error generating report: {str(e)}")
        return {
            "total_sessions": 0,
            "avg_session_duration_min": 0,
            "avg_questions_per_session": 0,
            "total_uploaded_files": 0,
            "most_common_questions": []
        }

def create_csv_report():
    """Create a comprehensive CSV report with multiple sheets"""
    try:
        report = generate_conversation_report()
        
        # Summary data
        summary_data = {
            'Metric': ['Total Sessions', 'Avg Duration (min)', 'Avg Questions/Session', 'Total Files'],
            'Value': [
                report['total_sessions'],
                report['avg_session_duration_min'],
                report['avg_questions_per_session'],
                report['total_uploaded_files']
            ]
        }
        
        # Questions data
        questions_data = {
            'Question': [q for q, _ in report['most_common_questions']],
            'Count': [count for _, count in report['most_common_questions']]
        }
        
        # Session data
        sessions = list(convo_collection.aggregate([
            {"$group": {
                "_id": "$session_id",
                "start_time": {"$min": "$timestamp"},
                "end_time": {"$max": "$timestamp"},
                "question_count": {"$sum": 1}
            }}
        ]))
        
        session_data = []
        for session in sessions:
            session_data.append({
                'Session ID': session['_id'],
                'Start Time': session['start_time'],
                'End Time': session['end_time'],
                'Duration (min)': round((session['end_time'] - session['start_time']).total_seconds() / 60, 2),
                'Question Count': session['question_count']
            })
        
        # Create DataFrames
        summary_df = pd.DataFrame(summary_data)
        questions_df = pd.DataFrame(questions_data)
        sessions_df = pd.DataFrame(session_data)
        
        # Create an Excel file with multiple sheets
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            questions_df.to_excel(writer, sheet_name='Top Questions', index=False)
            sessions_df.to_excel(writer, sheet_name='Sessions', index=False)
            
            # Add advanced analytics if available
            advanced_analysis = generate_advanced_analytics()
            if advanced_analysis and "categorized_questions" in advanced_analysis:
                category_data = {
                    'Category': [cat.replace('_', ' ').title() for cat in advanced_analysis["categorized_questions"].keys()],
                    'Count': list(advanced_analysis["categorized_questions"].values())
                }
                category_df = pd.DataFrame(category_data)
                category_df.to_excel(writer, sheet_name='Categories', index=False)
        
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error creating CSV report: {str(e)}")
        # Return a simple CSV as fallback
        report = generate_conversation_report()
        summary_data = {
            'Metric': ['Total Sessions', 'Avg Duration (min)', 'Avg Questions/Session', 'Total Files'],
            'Value': [
                report['total_sessions'],
                report['avg_session_duration_min'],
                report['avg_questions_per_session'],
                report['total_uploaded_files']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        return summary_df.to_csv(index=False).encode('utf-8')

def generate_advanced_analytics():
    """Generate advanced analytics with actionable insights and error handling"""
    try:
        # Get all conversations for analysis
        all_conversations = list(convo_collection.find({}))
        
        if not all_conversations:
            return {
                "basic_report": generate_conversation_report(),
                "categorized_questions": {},
                "category_examples": {},
                "insights": [],
                "total_questions_analyzed": 0
            }
        
        # Extract all questions
        all_questions = [conv.get("user_message", "").lower() for conv in all_conversations if "user_message" in conv]
        
        # Define question categories
        question_categories = {
            "grading_questions": ["grade", "mark", "score", "gpa", "pass", "fail", "minimum", "percent"],
            "exam_questions": ["exam", "test", "midterm", "final", "quiz", "assessment"],
            "course_questions": ["course", "subject", "credit", "prerequisite", "curriculum"],
            "registration_questions": ["register", "enroll", "add", "drop", "withdraw"],
            "financial_questions": ["fee", "payment", "scholarship", "financial aid", "tuition"],
            "procedural_questions": ["procedure", "process", "how to", "steps", "required"],
            "facility_questions": ["library", "lab", "classroom", "campus", "facility"],
            "staff_questions": ["professor", "instructor", "teacher", "advisor", "staff"]
        }
        
        # Categorize questions
        categorized_questions = {category: 0 for category in question_categories.keys()}
        category_examples = {category: [] for category in question_categories.keys()}
        
        for question in all_questions:
            categorized = False
            for category, keywords in question_categories.items():
                if any(keyword in question for keyword in keywords):
                    categorized_questions[category] += 1
                    if question not in category_examples[category]:
                        category_examples[category].append(question)
                    categorized = True
                    break
            
            # If not categorized, add to uncategorized
            if not categorized:
                if "uncategorized" not in categorized_questions:
                    categorized_questions["uncategorized"] = 0
                categorized_questions["uncategorized"] += 1
                if "uncategorized" not in category_examples:
                    category_examples["uncategorized"] = []
                category_examples["uncategorized"].append(question)
        
        # Generate insights
        insights = []
        total_questions = len(all_questions)
        
        # Check for high volume categories
        for category, count in categorized_questions.items():
            if count > 0 and total_questions > 0:
                percentage = (count / total_questions) * 100
                if percentage > 20:  # If more than 20% of questions fall into one category
                    insights.append({
                        "type": "info",
                        "title": f"📊 High Volume: {category.replace('_', ' ').title()}",
                        "message": f"{percentage:.1f}% of questions are about {category.replace('_', ' ')}",
                        "suggestion": f"Consider creating dedicated resources for {category.replace('_', ' ')}"
                    })
        
        # Check for time patterns
        try:
            # Get hourly distribution
            hour_counts = {}
            for conv in all_conversations:
                if "timestamp" in conv:
                    hour = conv["timestamp"].hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            if hour_counts:
                peak_hour = max(hour_counts, key=hour_counts.get)
                if hour_counts[peak_hour] > (len(all_conversations) / 24) * 2:  # More than twice the average
                    insights.append({
                        "type": "info",
                        "title": "⏰ Peak Usage Time",
                        "message": f"Most questions are asked around {peak_hour}:00 ({hour_counts[peak_hour]} questions)",
                        "suggestion": "Ensure adequate support resources are available during peak hours"
                    })
        except Exception as time_error:
            st.warning(f"Could not analyze time patterns: {time_error}")
        
        # Check for question complexity (length-based)
        question_lengths = [len(q) for q in all_questions]
        if question_lengths:
            avg_length = sum(question_lengths) / len(question_lengths)
            if avg_length > 50:
                insights.append({
                    "type": "info",
                    "title": "📝 Complex Questions",
                    "message": f"Average question length is {avg_length:.1f} characters",
                    "suggestion": "Users are asking detailed questions - ensure your knowledge base is comprehensive"
                })
        
        return {
            "basic_report": generate_conversation_report(),
            "categorized_questions": categorized_questions,
            "category_examples": category_examples,
            "insights": insights,
            "total_questions_analyzed": total_questions
        }
        
    except Exception as e:
        st.error(f"Error in advanced analytics: {str(e)}")
        # Return a basic report even if advanced analytics fails
        return {
            "basic_report": generate_conversation_report(),
            "categorized_questions": {},
            "category_examples": {},
            "insights": [],
            "total_questions_analyzed": 0
        }
    # Add this function after your existing helper functions:

def rebuild_file_database_from_vectorstore():
    """Rebuild file database records from existing vector store metadata"""
    try:
        if not faiss_index_exists():
            return "No vector store found to rebuild from"
        
        vectorstore = load_faiss()
        
        # Extract unique source files from vector store
        sources = set()
        doc_count = 0
        
        # Access documents through the docstore
        for doc_id in vectorstore.docstore._dict:
            doc = vectorstore.docstore._dict[doc_id]
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source and source != 'unknown':
                    sources.add(source)
                doc_count += 1
        
        # Create database records for missing files
        created_count = 0
        for source in sources:
            # Check if already exists
            existing = files_collection.find_one({"filename": source})
            if not existing:
                # Create a placeholder record (without actual file data)
                file_doc = {
                    "filename": source,
                    "file_id": None,  # No GridFS file, just metadata
                    "uploaded_at": datetime.now(),
                    "rebuilt_from_vector": True,
                    "note": "Rebuilt from vector store metadata - file data not available for download"
                }
                files_collection.insert_one(file_doc)
                created_count += 1
        
        return {
            "success": True,
            "total_docs_in_vector": doc_count,
            "unique_sources": len(sources),
            "created_records": created_count,
            "sources": list(sources)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Add this to your Vector Store Diagnostics section (in the admin area):

# === Enhanced Vector Store Check ===
if st.session_state.logged_in:
    with st.expander("🔍 Vector Store Diagnostics", expanded=False):
        st.markdown("### 🔍 Vector Store Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Check Vector Store Status"):
                try:
                    if faiss_index_exists():
                        vectorstore = load_faiss()
                        st.success(f"✅ Vector store loaded successfully")
                        st.write(f"Total documents: {vectorstore.index.ntotal}")
                        
                        # Test search
                        test_docs = vectorstore.similarity_search("grading policy", k=2)
                        st.write(f"Test search found {len(test_docs)} documents")
                        for i, doc in enumerate(test_docs):
                            st.write(f"**Document {i+1}:** {doc.page_content[:100]}...")
                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    else:
                        st.warning("⚠️ No vector store found. Please upload files first.")
                except Exception as e:
                    st.error(f"❌ Vector store error: {e}")
        
        with col2:
            if st.button("🔄 Rebuild Database Records"):
                with st.spinner("Rebuilding database from vector store..."):
                    result = rebuild_file_database_from_vectorstore()
                    
                    if isinstance(result, dict) and result.get("success"):
                        st.success(f"✅ Database rebuild complete!")
                        st.write(f"- Found {result['total_docs_in_vector']} documents in vector store")
                        st.write(f"- Identified {result['unique_sources']} unique source files")
                        st.write(f"- Created {result['created_records']} new database records")
                        
                        if result['sources']:
                            st.write("**Source files found:**")
                            for source in result['sources']:
                                st.write(f"  • {source}")
                    else:
                        st.error(f"❌ Rebuild failed: {result}")
        
        # Show sync status
        st.markdown("---")
        st.markdown("#### 🔄 Database-Vector Store Sync")
        
        # Count files in database
        db_count = files_collection.count_documents({})
        
        # Count vector store docs
        vector_count = 0
        vector_sources = set()
        if faiss_index_exists():
            try:
                vectorstore = load_faiss()
                vector_count = vectorstore.index.ntotal
                
                # Get unique sources
                for doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = doc.metadata['source']
                        if source and source != 'unknown':
                            vector_sources.add(source)
            except:
                pass
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Files", db_count)
        with col2:
            st.metric("Vector Documents", vector_count)
        with col3:
            st.metric("Unique Sources", len(vector_sources))
        
        if db_count == 0 and vector_count > 0:
            st.warning("⚠️ Vector store has documents but database has no file records. Click 'Rebuild Database Records' to fix this.")
        elif db_count > 0 and vector_count == 0:
            st.warning("⚠️ Database has file records but vector store is empty. Re-upload files to rebuild vector store.")
        elif db_count > 0 and vector_count > 0:
            st.success("✅ Both database and vector store contain data")
        else:
            st.info("ℹ️ Both database and vector store are empty. Upload files to get started.")



st.title(" Salasa go chatbot")

# === Sidebar Login ===
with st.sidebar:
    st.markdown("### 🔑 Admin Access")
    if st.session_state.logged_in:
        if st.button("Logout", type="primary"):
            st.session_state.logged_in = False
            st.session_state.admin_dashboard_expanded = False
            st.rerun()
        st.success("Logged in as admin")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
def check_and_fix_sync_issues():
    """Comprehensive sync check and fix function"""
    try:
        if not faiss_index_exists():
            return {
                "status": "error",
                "message": "No vector store found. Please upload files first."
            }
        
        # Get vector store sources
        vectorstore = load_faiss()
        vector_sources = set()
        
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
            for doc_id in vectorstore.docstore._dict:
                try:
                    doc = vectorstore.docstore._dict[doc_id]
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = doc.metadata['source']
                        if source and source != 'unknown':
                            vector_sources.add(source.strip())
                except:
                    continue
        
        # Get database files
        db_files = {f['filename'].strip() for f in files_collection.find({"file_id": {"$ne": None}}, {"filename": 1})}
        
        # Find mismatches
        missing_in_db = vector_sources - db_files
        missing_in_vector = db_files - vector_sources
        
        # Auto-fix missing database records
        fixed_count = 0
        if missing_in_db:
            for missing_file in missing_in_db:
                file_doc = {
                    "filename": missing_file,
                    "file_id": None,
                    "uploaded_at": datetime.now(),
                    "auto_created": True,
                    "status": "vector_only",
                    "note": "Auto-created to sync with vector store - original file not available for download"
                }
                files_collection.insert_one(file_doc)
                fixed_count += 1
        
        return {
            "status": "success",
            "vector_sources_count": len(vector_sources),
            "db_files_count": len(db_files),
            "missing_in_db": list(missing_in_db),
            "missing_in_vector": list(missing_in_vector),
            "auto_fixed": fixed_count,
            "is_synced": len(missing_in_db) == 0 and len(missing_in_vector) == 0
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error during sync check: {str(e)}"
        }

# === Admin Dashboard Section ===
if st.button(" Emergency Fix - Sync Database with Vector Store"):
    result = check_and_fix_sync_issues()
    if result["status"] == "success":
        st.success(f"✅ Fixed {result['auto_fixed']} missing database records!")
    else:
        st.error(f"❌ {result['message']}")

if st.session_state.logged_in:
    if st.button("⚙️ Admin Dashboard", type="primary"):
        st.session_state.admin_dashboard_expanded = not st.session_state.admin_dashboard_expanded
        st.rerun()
    
    if st.session_state.admin_dashboard_expanded:
        with st.expander("📊 Admin Dashboard", expanded=True):
            st.markdown("Chatbot Analytics")
            
            report = generate_conversation_report()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Sessions</div>
                    <div class="metric-value">{report['total_sessions']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Duration (min)</div>
                    <div class="metric-value">{report['avg_session_duration_min']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Questions/Session</div>
                    <div class="metric-value">{report['avg_questions_per_session']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Files</div>
                    <div class="metric-value">{report['total_uploaded_files']}</div>
                </div>
                """, unsafe_allow_html=True)
    
            if report['most_common_questions']:
                st.markdown("### 🔥 Most Common Questions")
                for i, (question, count) in enumerate(report['most_common_questions'], 1):
                    st.write(f"{i}. {question} ({count} times)")
            else:
                st.write("No questions recorded yet.")
            
            # CSV Download Button
            st.markdown("---")
            st.markdown("### 📊 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Generate CSV Report"):
                    csv_data = create_csv_report()
                    st.download_button(
                        label="📄 Download Excel Report",
                        data=csv_data,
                        file_name=f"almaarefa_chatbot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if st.button("📈 Generate Advanced Analytics"):
                    st.session_state.show_advanced_analytics = True
                    st.rerun()
            
            # Advanced Analytics Section
            if st.session_state.show_advanced_analytics:
                st.markdown("---")
                st.markdown("### 🔍 Advanced Analytics & Insights")
                
                with st.spinner("Analyzing conversation patterns..."):
                    advanced_analysis = generate_advanced_analytics()
                    
                    # Display insights
                    if advanced_analysis["insights"]:
                        st.markdown("#### 💡 Actionable Insights")
                        for insight in advanced_analysis["insights"]:
                            if insight["type"] == "warning":
                                st.warning(f"**{insight['title']}**\n\n{insight['message']}\n\n*Suggestion: {insight['suggestion']}*")
                            else:
                                st.info(f"**{insight['title']}**\n\n{insight['message']}\n\n*Suggestion: {insight['suggestion']}*")
                    else:
                        st.info("No significant patterns detected yet. More conversation data needed.")
                    
                    # Question categories
                    if advanced_analysis["categorized_questions"]:
                        st.markdown("#### 📊 Question Categories")
                        categories_df = pd.DataFrame({
                            'Category': [cat.replace('_', ' ').title() for cat in advanced_analysis["categorized_questions"].keys()],
                            'Count': list(advanced_analysis["categorized_questions"].values())
                        })
                        st.dataframe(categories_df.sort_values('Count', ascending=False))
                        
                        # Show chart
                        st.markdown("#### 📈 Category Distribution")
                        chart_data = categories_df.set_index('Category')
                        st.bar_chart(chart_data)
                    
                    st.write(f"*Analyzed {advanced_analysis['total_questions_analyzed']} total questions*")

# === Admin-only File Upload ===
if st.session_state.logged_in:
    with st.expander("📁 File Management (Admin Only)", expanded=False):
        st.markdown("### 📁 Upload University Documents")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Drag and drop university files here",
            type=["txt", "json", "pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload TXT, JSON, PDF, or image files containing university rules"
        )
        
        if uploaded_file:
            with st.spinner("Processing file..."):
                # Extract content
                content = load_content_with_ocr(uploaded_file)
                
                if content:
                    # Save file
                    save_result = save_uploaded_file_with_sync(uploaded_file)
                    
                    if save_result["success"]:
                        # Create documents and update vector store
                        docs = make_documents(content, source=uploaded_file.name)
                        load_faiss_or_create(docs)
                        
                        st.success(f"✅ {uploaded_file.name} uploaded and indexed successfully!")
                        st.info(f"📊 File saved with ID: {save_result['file_id']}")
                    else:
                        st.error(f"❌ Failed to save file: {save_result['error']}")
                else:
                    st.error("❌ Could not extract content from file")
        
        # Add a section to view and manage uploaded files
        st.markdown("---")
        st.markdown("### 📋 Uploaded Files")
        
        # Get all uploaded files
        uploaded_files = list(files_collection.find({}).sort("uploaded_at", -1))
        
        if uploaded_files:
            for file_record in uploaded_files:
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{file_record['filename']}**")
                    st.caption(f"Uploaded: {file_record['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Size: {file_record.get('file_size', 0) / 1024:.1f} KB")
                
                # In the File Management section, replace the download button code with:
                with col2:
                    # Download button
                    if st.button("⬇️ Download", key=f"download_{file_record['_id']}"):
                        if file_record:
                            download_result = get_file_for_download(file_record['filename'])
                            if download_result and isinstance(download_result, dict) and download_result.get("success"):
                                st.download_button(
                                    label="Click to download",
                                    data=download_result["data"],
                                    file_name=file_record['filename'],
                                    mime="application/octet-stream",
                                    key=f"dl_{file_record['_id']}"
                                )
                            else:
                                error_msg = download_result.get("message", "Unknown error") if download_result and isinstance(download_result, dict) else "Download failed"
                                st.error(f"❌ {error_msg}")
                        else:
                            st.error("❌ File record not found")
                
                with col3:
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{file_record['_id']}"):
                        # Delete from database
                        files_collection.delete_one({"_id": file_record["_id"]})
                        
                        # Delete from GridFS if exists
                        if file_record.get("file_id"):
                            try:
                                fs.delete(file_record["file_id"])
                            except:
                                pass
                        
                        # Delete local backup if exists
                        local_path = file_record.get("local_backup_path")
                        if local_path and os.path.exists(local_path):
                            try:
                                os.remove(local_path)
                            except:
                                pass
                        
                        # Rebuild vector store without this file
                        rebuild_vector_store_without_file(file_record['filename'])
                        
                        st.success(f"✅ {file_record['filename']} deleted successfully!")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No files uploaded yet.")


# Replace the old uploader section with this:
    
def handle_file_upload_with_proper_sync():
    """Handle file upload ensuring proper database-vector store sync"""
    uploaded_file = st.file_uploader(
        "Drag and drop university files here",
        type=["txt", "json", "pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload TXT, JSON, PDF, or image files containing university rules"
    )

    if uploaded_file:
        with st.spinner("Processing file..."):
            # Extract content
            content = load_content_with_ocr(uploaded_file)
            
            if content:
                # Save file with proper sync
                save_result = save_uploaded_file_with_sync(uploaded_file)
                
                if save_result["success"]:
                    # Create documents and update vector store
                    docs = make_documents(content, source=uploaded_file.name)
                    load_faiss_or_create(docs)
                    
                    st.success(f"✅ {uploaded_file.name} uploaded and indexed successfully!")
                    st.info(f"📊 File saved with ID: {save_result['file_id']}")
                    
                    # Verify sync immediately
                    verification = get_file_for_download(uploaded_file.name)
                    if verification["success"]:
                        st.success("✅ File is properly synced and available for download")
                    else:
                        st.warning(f"⚠️ Upload successful but sync issue detected: {verification['message']}")
                else:
                    st.error(f"❌ Failed to save file: {save_result['error']}")
            else:
                st.error("❌ Could not extract content from file")


# === Database Management Section ===
# In your Database Management section, add:
# Add this near the Database Management section
st.markdown("---")
st.markdown("#### 🚨 Emergency Fix Tools")

if st.button("🔄 Sync Database with GridFS"):
    with st.spinner("Checking for sync issues..."):
        result = check_and_fix_sync_issues()
        
    if result["status"] == "success":
        if result["fixed_count"] > 0:
            st.success(f"Fixed {result['fixed_count']} out of {result['total_broken']} files!")
        else:
            st.info("No sync issues found")
    else:
        st.error(f"Error during sync: {result['message']}")
if st.session_state.logged_in:
    with st.expander("🗄️ Database Management (Admin Only)", expanded=False):
        st.markdown("### 🗄️ Database Management")
        st.markdown("""
        <div class="db-section">
            <p>Manage the chatbot database and stored information</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clear Conversations", type="secondary"):
                result = convo_collection.delete_many({})
                st.success(f"✅ Cleared {result.deleted_count} conversations")
                
        with col2:
            if st.button("🧹 Clear Files", type="secondary"):
                result = files_collection.delete_many({})
                st.success(f"✅ Cleared {result.deleted_count} file records")
                
        with col3:
            if st.button("🔄 Reset Vector Store", type="secondary"):
                if os.path.exists(INDEX_DIR):
                    import shutil
                    shutil.rmtree(INDEX_DIR)
                    st.success("✅ Vector store reset")
                else:
                    st.info("ℹ️ No vector store found to reset")
        
        st.markdown("---")
        st.markdown("""
        <div class="danger-zone">
            <h3>⚠️ Danger Zone</h3>
            <p>These actions cannot be undone. Proceed with caution.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💥 Reset Entire Database", type="primary", key="reset_db"):
            # Clear conversations
            convo_result = convo_collection.delete_many({})
            # Clear files
            file_result = files_collection.delete_many({})
            # Reset vector store
            if os.path.exists(INDEX_DIR):
                import shutil
                shutil.rmtree(INDEX_DIR)
            
            st.success(f"✅ Database reset complete! Removed {convo_result.deleted_count} conversations and {file_result.deleted_count} files")

# === Vector Store Check ===
if st.session_state.logged_in:
    with st.expander("🔍 Vector Store Diagnostics", expanded=False):
        st.markdown("### 🔍 Vector Store Status")
        
        if st.button("Check Vector store Status"):
            try:
                if faiss_index_exists():
                    vectorstore = load_faiss()
                    st.success(f"✅ Vector store loaded successfully")
                    st.write(f"Total documents: {vectorstore.index.ntotal}")
                    
                    # Test search
                    test_docs = vectorstore.similarity_search("grading policy", k=2)
                    st.write(f"Test search found {len(test_docs)} documents")
                    for i, doc in enumerate(test_docs):
                        st.write(f"**Document {i+1}:** {doc.page_content[:100]}...")
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                else:
                    st.warning("⚠️ No vector store found. Please upload files first.")
            except Exception as e:
                st.error(f"❌ Vector store error: {e}")

# Replace the User QA Section (starting around line 450) with this corrected version:

# === User QA Section ===
st.markdown("---")
st.markdown("### 💬 Ask a Question")

# Initialize used_files outside the conditional block
used_files = set()

if faiss_index_exists():
    user_question = st.text_input(
        "Welcome to the Educational Website Chatbot! Ask me anything about the site content:",
        key="typed_question",
        placeholder="e.g., How can I access the online courses?"
    )

    if user_question.strip():
        vectorstore = load_faiss()
        # Get only the most relevant document (k=1)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        # Detect language more robustly
        try:
            lang = detect(user_question)
            st.session_state.current_language = 'arabic' if lang.startswith('ar') else 'english'
        except Exception:
            lang = "en"
            st.session_state.current_language = 'english'
        
        # Enhanced Arabic prompt with explicit response language instruction
        if st.session_state.current_language == 'arabic':
            template_text = """استنادًا إلى المعلومات التالية من قاعدة بيانات الموقع التعليمي:

{context}

الآن أجب على السؤال التالي بدقة ووضوح باللغة العربية فقط. قدم الإجابة المحددة فقط التي تتعلق مباشرة بالسؤال. لا تذكر جميع المعلومات، فقط المعلومة ذات الصلة المباشرة:

{question}

إذا لم تجد المعلومات المطلوبة في السياق المعطى، فقل ذلك بوضوح باللغة العربية.

يجب أن تكون الإجابة باللغة العربية فقط."""
        else:
            template_text = """Based on the following information from the educational website database:

{context}

Now answer the following question accurately and clearly in English. Provide only the specific answer that directly relates to the question. Do not include all the information — only what is directly relevant:

{question}

If the required information is not found in the given context, please state that clearly in English.

You must respond in English only."""

        # Get only the most relevant document
        docs = retriever.get_relevant_documents(user_question)

        
        if docs:
            # Use only the most relevant document
            doc = docs[0]
            context = f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            used_files = {doc.metadata.get("source", "unknown").strip()}
        else:
            context = "No relevant information found in the database."
            used_files = set()
        
        prompt = PromptTemplate(
            input_variables=["context", "question"], 
            template=template_text
        )
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        with st.spinner("🤖 Thinking..." if st.session_state.current_language == 'english' else "🤖 جاري التفكير..."):
            answer = llm_chain.run({
                "context": context,
                "question": user_question
            })
            
        # Post-process the answer to ensure it's in the correct language
        if st.session_state.current_language == 'arabic':
            # Check if answer is in Arabic, if not, add a prefix
            try:
                answer_lang = detect(answer)
                if not answer_lang.startswith('ar'):
                    answer = "باللغة العربية: " + answer
            except:
                answer = "باللغة العربية: " + answer
        
        save_conversation(user_question, answer)
        
        # Display conversation in styled format
        st.markdown("---")
        st.markdown("### 📝 Conversation")
        
        # User message
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">
                <span>👤 You</span>
                <span>{datetime.now().strftime('%H:%M')}</span>
            </div>
            <p>{user_question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
        <div class="bot-message">
            <div class="message-header">
                <span>🤖 Assistant</span>
                <span>{datetime.now().strftime('%H:%M')}</span>
            </div>
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Language indicator
        language_indicator = "🌍 اللغة: العربية" if st.session_state.current_language == 'arabic' else "🌍 Language: English"
        st.info(language_indicator)
        
        # Replace the SOURCE FILES DOWNLOAD SECTION with this enhanced version that provides better debugging:

        # === SOURCE FILES DOWNLOAD SECTION ===# Replace the entire SOURCE FILES DOWNLOAD SECTION with:
if used_files:
    st.markdown("### 📂 Source File Used")
    st.info("You can download the source document that was used to generate this answer:")
    
    for source_file in used_files:
        if source_file != "unknown":
            # Debug: Show what we're searching for (always show this info)
            st.write(f"🔍 Looking for file: **{source_file}**")
            
            # Check what's actually in the database
            all_files = list(files_collection.find({}, {"filename": 1, "file_id": 1, "uploaded_at": 1}))
            st.write(f"📁 Available files in database: {len(all_files)}")
            
            if all_files:
                st.write("**Database files:**")
                for f in all_files:
                    st.write(f"  • {f['filename']} (ID: {str(f['file_id'])[:8]}...)")
            
            # Try multiple search strategies
            file_record = None
            search_method = None
            
            # Strategy 1: Exact match
            file_record = files_collection.find_one({"filename": source_file})
            if file_record:
                search_method = "exact match"
            
            # Strategy 2: Case-insensitive exact match
            if not file_record:
                file_record = files_collection.find_one({
                    "filename": {"$regex": f"^{re.escape(source_file.strip())}$", "$options": "i"}
                })
                if file_record:
                    search_method = "case-insensitive exact"
            
            # Strategy 3: Partial match
            if not file_record:
                file_record = files_collection.find_one({
                    "filename": {"$regex": re.escape(source_file.strip()), "$options": "i"}
                })
                if file_record:
                    search_method = "partial match"
            
            # Strategy 4: Try without file extension if source_file has one
            if not file_record and '.' in source_file:
                base_name = source_file.rsplit('.', 1)[0]
                file_record = files_collection.find_one({
                    "filename": {"$regex": re.escape(base_name), "$options": "i"}
                })
                if file_record:
                    search_method = "base name match"
            
            # Strategy 5: Try with common extensions if source_file doesn't have one
            if not file_record and '.' not in source_file:
                for ext in ['.txt', '.pdf', '.json']:
                    file_record = files_collection.find_one({
                        "filename": {"$regex": f"^{re.escape(source_file.strip())}{re.escape(ext)}$", "$options": "i"}
                    })
                    if file_record:
                        search_method = f"with {ext} extension"
                        break
            
            if file_record:
                st.success(f"✅ Found file using: {search_method}")
                st.write(f"**Matched file:** {file_record['filename']}")
                
                download_result = get_file_for_download(file_record['filename'])
                if download_result and isinstance(download_result, dict) and download_result.get("success"):
                    st.download_button(
                        label=f"⬇️ Download {file_record['filename']}",
                        data=download_result["data"],
                        file_name=file_record['filename'],
                        mime="application/octet-stream",
                        key=f"download_{file_record['_id']}"
                    )
                    st.success("✅ File is ready for download!")
                else:
                    error_msg = download_result.get("message", "Unknown error") if download_result and isinstance(download_result, dict) else "Download failed"
                    st.error(f"❌ {error_msg}")
                    
            else:
                st.error(f"❌ Could not find file: **{source_file}**")
                st.write("**Possible causes:**")
                st.write("1. File name mismatch between vector store and database")
                st.write("2. File was not properly saved during upload")  
                st.write("3. Database and vector store are out of sync")
                
                if st.session_state.logged_in:
                    st.write("**Admin suggestions:**")
                    st.write("- Re-upload the file to ensure proper storage")
                    st.write("- Check vector store diagnostics")
                    st.write("- Consider rebuilding the vector store")
        else:
            st.info("📄 Source file name not available")

# Also add this debugging function to the admin section:
def sync_database_with_vectorstore():
    """Check and fix sync issues between database and vector store"""
    try:
        if not faiss_index_exists():
            return "No vector store found"
        
        vectorstore = load_faiss()
        
        # Get all document sources from vector store
        vector_sources = set()
        for i in range(vectorstore.index.ntotal):
            try:
                doc = vectorstore.docstore._dict[list(vectorstore.docstore._dict.keys())[i]]
                if 'source' in doc.metadata:
                    vector_sources.add(doc.metadata['source'])
            except:
                continue
        
        # Get all files from database
        db_files = {f['filename'] for f in files_collection.find({}, {"filename": 1})}
        
        # Find mismatches
        missing_in_db = vector_sources - db_files
        missing_in_vector = db_files - vector_sources
        
        result = {
            "vector_sources": len(vector_sources),
            "db_files": len(db_files),
            "missing_in_db": list(missing_in_db),
            "missing_in_vector": list(missing_in_vector),
            "in_sync": len(missing_in_db) == 0 and len(missing_in_vector) == 0
        }
        
        return result
        
    except Exception as e:
        return f"Error checking sync: {str(e)}"
# === Email Conversation Option ===
if st.button("💾 Email Conversation History"):
    current_session_convos = list(convo_collection.find(
        {"session_id": st.session_state.session_id}
    ).sort("timestamp", 1))
    
    if current_session_convos:
        conversation_text = " Conversation History\n\n"
        for convo in current_session_convos:
            conversation_text += f"You: {convo['user_message']}\n"
            conversation_text += f"Assistant: {convo['bot_response']}\n"
            conversation_text += f"Time: {convo['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        user_email = st.text_input("Enter your email to receive the conversation:")
        if st.button("Send Email"):
            if user_email:
                result = send_conversation_email(user_email, conversation_text)
                if result is True:
                    st.success("✅ Conversation history sent successfully!")
                else:
                    st.error(f"❌ Failed to send email: {result}")
            else:
                st.warning("Please enter a valid email address.")
    else:
        st.warning("No conversation history to email.")

