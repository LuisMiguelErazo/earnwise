import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Definir el diccionario industry_info
industry_info = {
    "Engineering": {
        "soft_skills": [
            "8. Problem-solving", "7. Creativity", "6. Attention to detail", "5. Time management",
            "4. Adaptability", "3. Critical thinking", "2. Teamwork", "1. Communication skills"
        ],
        "software_programs": [
            "AutoCAD", "MATLAB", "SolidWorks", "Python", "C++", "Microsoft Office Suite", "LabVIEW"
        ],
        "study_fields": [
            "Engineering (Mechanical, Electrical, Civil, Software)", "Computer Science", "Applied Mathematics", "Physics"
        ]
    },
    "Analysis": {
        "soft_skills": [
            "8. Analytical thinking", "7. Attention to detail", "6. Problem-solving", "5. Critical thinking",
            "4. Communication skills", "3. Data interpretation", "2. Time management", "1. Research skills"
        ],
        "software_programs": [
            "SQL", "Python", "R", "Excel", "Tableau", "Power BI", "SAS"
        ],
        "study_fields": [
            "Data Science", "Statistics", "Business Analytics", "Economics", "Mathematics"
        ]
    },
    "Consulting": {
        "soft_skills": [
            "8. Communication skills", "7. Problem-solving", "6. Analytical thinking", "5. Client relationship management",
            "4. Adaptability", "3. Time management", "2. Presentation skills", "1. Negotiation skills"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Tableau", "Power BI", "SQL", "CRM software"
        ],
        "study_fields": [
            "Business Administration", "Management", "Economics", "Finance", "Psychology"
        ]
    },
    "Management": {
        "soft_skills": [
            "8. Leadership", "7. Communication skills", "6. Decision-making", "5. Problem-solving",
            "4. Time management", "3. Strategic thinking", "2. Teamwork", "1. Conflict resolution"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Tableau", "Power BI", "Slack", "Asana", "Trello"
        ],
        "study_fields": [
            "Business Administration", "Management", "Organizational Behavior", "Human Resources", "Operations Management"
        ]
    },
    "Sales": {
        "soft_skills": [
            "8. Communication skills", "7. Persuasion", "6. Negotiation skills", "5. Customer service",
            "4. Relationship building", "3. Time management", "2. Problem-solving", "1. Adaptability"
        ],
        "software_programs": [
            "Salesforce", "HubSpot", "CRM software", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Business Administration", "Marketing", "Communication", "Economics", "Psychology"
        ]
    },
    "Support": {
        "soft_skills": [
            "8. Communication skills", "7. Patience", "6. Problem-solving", "5. Empathy",
            "4. Customer service", "3. Adaptability", "2. Attention to detail", "1. Time management"
        ],
        "software_programs": [
            "CRM software", "Microsoft Office Suite", "Zendesk", "Freshdesk"
        ],
        "study_fields": [
            "Communication", "Business Administration", "Information Technology", "Psychology"
        ]
    },
    "Healthcare": {
        "soft_skills": [
            "8. Communication skills", "7. Empathy", "6. Attention to detail", "5. Problem-solving",
            "4. Teamwork", "3. Time management", "2. Adaptability", "1. Patience"
        ],
        "software_programs": [
            "Electronic Health Records (EHR) software", "Microsoft Office Suite", "Medical imaging software"
        ],
        "study_fields": [
            "Medicine", "Nursing", "Healthcare Administration", "Psychology", "Biomedical Sciences"
        ]
    },
    "Education": {
        "soft_skills": [
            "8. Communication skills", "7. Patience", "6. Empathy", "5. Organizational skills",
            "4. Problem-solving", "3. Adaptability", "2. Creativity", "1. Time management"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Learning Management Systems (LMS)", "Google Classroom"
        ],
        "study_fields": [
            "Education", "Educational Psychology", "Curriculum and Instruction", "Special Education", "Educational Administration"
        ]
    },
    "Finance": {
        "soft_skills": [
            "8. Analytical thinking", "7. Attention to detail", "6. Problem-solving", "5. Communication skills",
            "4. Ethical judgment", "3. Time management", "2. Critical thinking", "1. Decision-making"
        ],
        "software_programs": [
            "Excel", "SQL", "Python", "Financial modeling software", "Bloomberg Terminal"
        ],
        "study_fields": [
            "Finance", "Accounting", "Economics", "Business Administration", "Mathematics"
        ]
    },
    "Marketing": {
        "soft_skills": [
            "8. Creativity", "7. Communication skills", "6. Analytical thinking", "5. Problem-solving",
            "4. Adaptability", "3. Strategic thinking", "2. Teamwork", "1. Time management"
        ],
        "software_programs": [
            "Salesforce", "HubSpot", "Google Analytics", "SEO tools", "CRM software"
        ],
        "study_fields": [
            "Marketing", "Business Administration", "Communication", "Public Relations", "Digital Marketing"
        ]
    },
    "Human Resources": {
        "soft_skills": [
            "8. Communication skills", "7. Empathy", "6. Problem-solving", "5. Conflict resolution",
            "4. Adaptability", "3. Organizational skills", "2. Time management", "1. Negotiation skills"
        ],
        "software_programs": [
            "HR Information Systems (HRIS)", "Applicant Tracking Systems (ATS)", "Payroll software", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Human Resources Management", "Business Administration", "Psychology", "Organizational Behavior", "Labor Relations"
        ]
    },
    "Information Technology": {
        "soft_skills": [
            "8. Analytical thinking", "7. Problem-solving", "6. Communication skills", "5. Attention to detail",
            "4. Time management", "3. Adaptability", "2. Teamwork", "1. Technical proficiency"
        ],
        "software_programs": [
            "SQL", "Python", "Java", "Linux/Unix", "Microsoft Office Suite", "Network management tools",
            "Cybersecurity tools (e.g., firewalls, antivirus software)", "Database management systems (e.g., Oracle, MySQL)"
        ],
        "study_fields": [
            "Computer Science", "Information Technology", "Cybersecurity", "Software Engineering", "Network Administration"
        ]
    },
    "Legal": {
        "soft_skills": [
            "8. Analytical thinking", "7. Attention to detail", "6. Communication skills", "5. Problem-solving",
            "4. Ethical judgment", "3. Time management", "2. Critical thinking", "1. Negotiation skills"
        ],
        "software_programs": [
            "Legal research databases (e.g., LexisNexis, Westlaw)", "Case management software", "Document management systems",
            "Microsoft Office Suite"
        ],
        "study_fields": [
            "Law", "Legal Studies", "Political Science", "Criminal Justice", "Business Law"
        ]
    },
    "Operations": {
        "soft_skills": [
            "8. Problem-solving", "7. Analytical thinking", "6. Communication skills", "5. Organizational skills",
            "4. Time management", "3. Teamwork", "2. Adaptability", "1. Decision-making"
        ],
        "software_programs": [
            "ERP systems (e.g., SAP, Oracle)", "Microsoft Office Suite", "Project management software (e.g., Asana, Trello)",
            "Data analysis tools (e.g., Excel, Tableau)"
        ],
        "study_fields": [
            "Operations Management", "Business Administration", "Supply Chain Management", "Industrial Engineering", "Logistics"
        ]
    },
    "Product Management": {
        "soft_skills": [
            "8. Communication skills", "7. Problem-solving", "6. Analytical thinking", "5. Project management",
            "4. Time management", "3. Teamwork", "2. Strategic thinking", "1. Decision-making"
        ],
        "software_programs": [
            "JIRA", "Asana", "Trello", "Microsoft Office Suite", "Product lifecycle management (PLM) tools",
            "Analytics tools (e.g., Google Analytics, Tableau)"
        ],
        "study_fields": [
            "Business Administration", "Marketing", "Computer Science", "Engineering", "Economics"
        ]
    },
    "Project Management": {
        "soft_skills": [
            "8. Leadership", "7. Communication skills", "6. Problem-solving", "5. Time management",
            "4. Teamwork", "3. Organizational skills", "2. Decision-making", "1. Negotiation skills"
        ],
        "software_programs": [
            "Microsoft Project", "Asana", "Trello", "JIRA", "Slack", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Project Management", "Business Administration", "Engineering", "Information Technology", "Operations Management"
        ]
    },
    "Research": {
        "soft_skills": [
            "8. Analytical thinking", "7. Attention to detail", "6. Problem-solving", "5. Communication skills",
            "4. Time management", "3. Critical thinking", "2. Adaptability", "1. Research skills"
        ],
        "software_programs": [
            "Statistical analysis software (e.g., SPSS, R)", "Data visualization tools (e.g., Tableau, Power BI)",
            "Microsoft Office Suite", "Research databases", "Lab management software"
        ],
        "study_fields": [
            "Research Methodology", "Statistics", "Data Science", "Biology", "Chemistry"
        ]
    },
    "Supply Chain": {
        "soft_skills": [
            "8. Organizational skills", "7. Problem-solving", "6. Communication skills", "5. Time management",
            "4. Analytical thinking", "3. Attention to detail", "2. Teamwork", "1. Negotiation skills"
        ],
        "software_programs": [
            "ERP systems (e.g., SAP, Oracle)", "Microsoft Office Suite", "Supply chain management software",
            "Inventory management systems", "Data analysis tools (e.g., Excel)"
        ],
        "study_fields": [
            "Supply Chain Management", "Logistics", "Operations Management", "Industrial Engineering", "Business Administration"
        ]
    },
    "Training and Development": {
        "soft_skills": [
            "8. Communication skills", "7. Patience", "6. Problem-solving", "5. Organizational skills",
            "4. Adaptability", "3. Creativity", "2. Leadership", "1. Time management"
        ],
        "software_programs": [
            "Learning Management Systems (LMS)", "Microsoft Office Suite", "Presentation software (e.g., PowerPoint, Prezi)",
            "E-learning software (e.g., Articulate, Captivate)"
        ],
        "study_fields": [
            "Human Resources Management", "Education", "Organizational Development", "Psychology", "Business Administration"
        ]
    },
    "Writing and Editing": {
        "soft_skills": [
            "8. Communication skills", "7. Attention to detail", "6. Creativity", "5. Time management",
            "4. Critical thinking", "3. Adaptability", "2. Research skills", "1. Teamwork"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Google Docs", "Grammarly", "Content management systems (CMS)", "Adobe Creative Suite"
        ],
        "study_fields": [
            "English", "Journalism", "Communications", "Creative Writing", "Marketing"
        ]
    },
    "Construction": {
        "soft_skills": [
            "8. Teamwork", "7. Communication skills", "6. Problem-solving", "5. Attention to detail",
            "4. Time management", "3. Adaptability", "2. Physical stamina", "1. Technical proficiency"
        ],
        "software_programs": [
            "AutoCAD", "Revit", "Microsoft Office Suite", "Project management software (e.g., Procore)", "Estimation software (e.g., Bluebeam)"
        ],
        "study_fields": [
            "Construction Management", "Civil Engineering", "Architecture", "Project Management", "Industrial Technology"
        ]
    },
    "Retail": {
        "soft_skills": [
            "8. Customer service", "7. Communication skills", "6. Problem-solving", "5. Teamwork",
            "4. Time management", "3. Adaptability", "2. Organizational skills", "1. Attention to detail"
        ],
        "software_programs": [
            "Point of Sale (POS) systems", "Inventory management software", "Microsoft Office Suite", "CRM software"
        ],
        "study_fields": [
            "Business Administration", "Marketing", "Retail Management", "Communications", "Sales"
        ]
    },
    "Customer Service": {
        "soft_skills": [
            "8. Communication skills", "7. Patience", "6. Problem-solving", "5. Empathy",
            "4. Customer service", "3. Adaptability", "2. Attention to detail", "1. Time management"
        ],
        "software_programs": [
            "CRM software", "Help desk software (e.g., Zendesk, Freshdesk)", "Microsoft Office Suite", "Call center software"
        ],
        "study_fields": [
            "Communications", "Business Administration", "Psychology", "Customer Service Management", "Marketing"
        ]
    },
    "Administrative": {
        "soft_skills": [
            "8. Organizational skills", "7. Communication skills", "6. Attention to detail", "5. Time management",
            "4. Problem-solving", "3. Adaptability", "2. Teamwork", "1. Customer service"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Google Workspace", "Scheduling software", "Document management systems"
        ],
        "study_fields": [
            "Business Administration", "Office Administration", "Communication", "Management", "Public Administration"
        ]
    },
    "Food and Beverages": {
        "soft_skills": [
            "8. Communication skills", "7. Teamwork", "6. Attention to detail", "5. Time management",
            "4. Adaptability", "3. Customer service", "2. Problem-solving", "1. Creativity"
        ],
        "software_programs": [
            "Point of Sale (POS) systems", "Inventory management software", "Scheduling software", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Culinary Arts", "Hospitality Management", "Food Science", "Business Administration", "Nutrition"
        ]
    },
    "Manufacturing": {
        "soft_skills": [
            "8. Attention to detail", "7. Problem-solving", "6. Teamwork", "5. Time management",
            "4. Adaptability", "3. Communication skills", "2. Technical proficiency", "1. Organizational skills"
        ],
        "software_programs": [
            "ERP systems", "Microsoft Office Suite", "CAD software", "Quality control software"
        ],
        "study_fields": [
            "Industrial Engineering", "Manufacturing Engineering", "Mechanical Engineering", "Business Administration", "Quality Assurance"
        ]
    },
    "Logistics": {
        "soft_skills": [
            "8. Organizational skills", "7. Problem-solving", "6. Communication skills", "5. Time management",
            "4. Analytical thinking", "3. Attention to detail", "2. Teamwork", "1. Negotiation skills"
        ],
        "software_programs": [
            "ERP systems (e.g., SAP, Oracle)", "Microsoft Office Suite", "Supply chain management software",
            "Inventory management systems", "Data analysis tools (e.g., Excel)"
        ],
        "study_fields": [
            "Supply Chain Management", "Logistics", "Operations Management", "Industrial Engineering", "Business Administration"
        ]
    },
    "Arts and Design": {
        "soft_skills": [
            "8. Creativity", "7. Communication skills", "6. Attention to detail", "5. Time management",
            "4. Adaptability", "3. Problem-solving", "2. Teamwork", "1. Critical thinking"
        ],
        "software_programs": [
            "Adobe Creative Suite", "AutoCAD", "SketchUp", "Microsoft Office Suite", "Digital design tools (e.g., Figma, InVision)"
        ],
        "study_fields": [
            "Graphic Design", "Fine Arts", "Interior Design", "Architecture", "Industrial Design"
        ]
    },
    "Entertainment": {
        "soft_skills": [
            "8. Creativity", "7. Communication skills", "6. Teamwork", "5. Adaptability",
            "4. Problem-solving", "3. Time management", "2. Presentation skills", "1. Networking skills"
        ],
        "software_programs": [
            "Adobe Creative Suite", "Final Cut Pro", "Avid Media Composer", "Microsoft Office Suite", "Sound editing software (e.g., Pro Tools)"
        ],
        "study_fields": [
            "Film and Television Production", "Performing Arts", "Music", "Media Studies", "Communication"
        ]
    },
    "Real Estate": {
        "soft_skills": [
            "8. Communication skills", "7. Negotiation skills", "6. Customer service", "5. Problem-solving",
            "4. Attention to detail", "3. Time management", "2. Adaptability", "1. Sales skills"
        ],
        "software_programs": [
            "CRM software", "Microsoft Office Suite", "Property management software", "MLS (Multiple Listing Service) platforms"
        ],
        "study_fields": [
            "Real Estate", "Business Administration", "Marketing", "Finance", "Urban Planning"
        ]
    },
    "Healthcare Support": {
        "soft_skills": [
            "8. Communication skills", "7. Attention to detail", "6. Empathy", "5. Customer service",
            "4. Problem-solving", "3. Teamwork", "2. Adaptability", "1. Time management"
        ],
        "software_programs": [
            "Electronic Health Records (EHR) software", "Microsoft Office Suite", "Medical billing and coding software"
        ],
        "study_fields": [
            "Healthcare Administration", "Medical Billing and Coding", "Health Information Management", "Nursing", "Medical Assisting"
        ]
    },
    "Human Services": {
        "soft_skills": [
            "8. Empathy", "7. Communication skills", "6. Problem-solving", "5. Teamwork",
            "4. Time management", "3. Adaptability", "2. Conflict resolution", "1. Patience"
        ],
        "software_programs": [
            "Case management software", "Microsoft Office Suite", "Database management systems"
        ],
        "study_fields": [
            "Social Work", "Psychology", "Counseling", "Human Services", "Sociology"
        ]
    },
    "Fitness": {
        "soft_skills": [
            "8. Communication skills", "7. Motivation", "6. Problem-solving", "5. Patience",
            "4. Adaptability", "3. Teamwork", "2. Time management", "1. Customer service"
        ],
        "software_programs": [
            "Fitness tracking apps", "Scheduling software", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Kinesiology", "Exercise Science", "Sports Management", "Physical Education", "Nutrition"
        ]
    },
    "Aviation and Aerospace": {
        "soft_skills": [
            "8. Attention to detail", "7. Problem-solving", "6. Communication skills", "5. Teamwork",
            "4. Adaptability", "3. Time management", "2. Technical proficiency", "1. Critical thinking"
        ],
        "software_programs": [
            "Flight simulation software", "Aircraft maintenance software", "Microsoft Office Suite", "CAD software (e.g., CATIA, SolidWorks)"
        ],
        "study_fields": [
            "Aerospace Engineering", "Aviation Management", "Mechanical Engineering", "Electrical Engineering", "Pilot Training"
        ]
    },
    "Automotive": {
        "soft_skills": [
            "8. Problem-solving", "7. Technical proficiency", "6. Communication skills", "5. Attention to detail",
            "4. Time management", "3. Teamwork", "2. Customer service", "1. Adaptability"
        ],
        "software_programs": [
            "Diagnostic tools (e.g., OBD-II scanners)", "AutoCAD", "Microsoft Office Suite"
        ],
        "study_fields": [
            "Automotive Technology", "Mechanical Engineering", "Electrical Engineering", "Business Administration", "Industrial Technology"
        ]
    },
    "Banking": {
        "soft_skills": [
            "8. Communication skills", "7. Customer service", "6. Attention to detail", "5. Problem-solving",
            "4. Analytical thinking", "3. Time management", "2. Teamwork", "1. Ethical judgment"
        ],
        "software_programs": [
            "Banking software", "Microsoft Office Suite", "Financial analysis tools"
        ],
        "study_fields": [
            "Finance", "Accounting", "Business Administration", "Economics", "Banking"
        ]
    },
    "Biotechnology": {
        "soft_skills": [
            "8. Analytical thinking", "7. Problem-solving", "6. Attention to detail", "5. Communication skills",
            "4. Research skills", "3. Teamwork", "2. Time management", "1. Adaptability"
        ],
        "software_programs": [
            "Laboratory information management systems (LIMS)", "Bioinformatics software", "Microsoft Office Suite",
            "Statistical analysis software (e.g., SPSS, R)"
        ],
        "study_fields": [
            "Biotechnology", "Molecular Biology", "Biochemistry", "Biomedical Engineering", "Genetics"
        ]
    },
    "Communications": {
        "soft_skills": [
            "8. Communication skills", "7. Public speaking", "6. Problem-solving", "5. Creativity",
            "4. Adaptability", "3. Teamwork", "2. Time management", "1. Conflict resolution"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Social media management tools", "CRM software", "Content management systems (CMS)"
        ],
        "study_fields": [
            "Communications", "Public Relations", "Journalism", "Marketing", "Media Studies"
        ]
    },
    "Energy and Utilities": {
        "soft_skills": [
            "8. Technical proficiency", "7. Safety awareness", "6. Analytical thinking", "5. Problem-solving",
            "4. Attention to detail", "3. Communication skills", "2. Critical thinking", "1. Teamwork"
        ],
        "software_programs": [
            "SCADA systems", "Microsoft Office Suite", "GIS software", "Energy management systems"
        ],
        "study_fields": [
            "Energy Management", "Environmental Science", "Electrical Engineering", "Mechanical Engineering", "Renewable Energy"
        ]
    },
    "Environmental Services": {
        "soft_skills": [
            "8. Analytical thinking", "7. Research skills", "6. Attention to detail", "5. Problem-solving",
            "4. Communication skills", "3. Critical thinking", "2. Teamwork", "1. Adaptability"
        ],
        "software_programs": [
            "GIS software", "Microsoft Office Suite", "Environmental monitoring software", "Data analysis tools"
        ],
        "study_fields": [
            "Environmental Science", "Ecology", "Environmental Engineering", "Geology", "Biology"
        ]
    },
    "Hospitality": {
        "soft_skills": [
            "8. Customer service", "7. Communication skills", "6. Problem-solving", "5. Teamwork",
            "4. Time management", "3. Adaptability", "2. Organizational skills", "1. Attention to detail"
        ],
        "software_programs": [
            "Property management systems (PMS)", "Event management software", "Microsoft Office Suite", "Reservation systems"
        ],
        "study_fields": [
            "Hospitality Management", "Tourism", "Business Administration", "Event Planning", "Hotel Management"
        ]
    },
    "Insurance": {
        "soft_skills": [
            "8. Communication skills", "7. Customer service", "6. Problem-solving", "5. Analytical thinking",
            "4. Attention to detail", "3. Negotiation skills", "2. Time management", "1. Teamwork"
        ],
        "software_programs": [
            "CRM software", "Microsoft Office Suite", "Insurance claims management software", "Actuarial software"
        ],
        "study_fields": [
            "Finance", "Accounting", "Business Administration", "Actuarial Science", "Risk Management"
        ]
    },
    "Media and Entertainment": {
        "soft_skills": [
            "8. Creativity", "7. Communication skills", "6. Teamwork", "5. Adaptability",
            "4. Problem-solving", "3. Time management", "2. Presentation skills", "1. Networking skills"
        ],
        "software_programs": [
            "Adobe Creative Suite", "Final Cut Pro", "Avid Media Composer", "Microsoft Office Suite", "Sound editing software (e.g., Pro Tools)"
        ],
        "study_fields": [
            "Film and Television Production", "Performing Arts", "Music", "Media Studies", "Communication"
        ]
    },
    "Public Sector": {
        "soft_skills": [
            "8. Communication skills", "7. Problem-solving", "6. Teamwork", "5. Adaptability",
            "4. Analytical thinking", "3. Attention to detail", "2. Organizational skills", "1. Time management"
        ],
        "software_programs": [
            "Microsoft Office Suite", "Data analysis tools", "Project management software", "Policy analysis tools"
        ],
        "study_fields": [
            "Public Administration", "Political Science", "Sociology", "Public Policy", "Economics"
        ]
    },
    "Telecommunications": {
        "soft_skills": [
            "8. Technical proficiency", "7. Communication skills", "6. Problem-solving", "5. Attention to detail",
            "4. Teamwork", "3. Adaptability", "2. Time management", "1. Customer service"
        ],
        "software_programs": [
            "Network management tools", "Microsoft Office Suite", "Telecommunications software", "CAD software"
        ],
        "study_fields": [
            "Telecommunications Engineering", "Electrical Engineering", "Computer Science", "Information Technology", "Network Administration"
        ]
    },
    "Transportation": {
        "soft_skills": [
            "8. Communication skills", "7. Time management", "6. Problem-solving", "5. Attention to detail",
            "4. Teamwork", "3. Adaptability", "2. Customer service", "1. Organizational skills"
        ],
        "software_programs": [
            "Fleet management software", "GPS tracking systems", "Microsoft Office Suite", "Logistics software"
        ],
        "study_fields": [
            "Transportation Management", "Logistics", "Supply Chain Management", "Business Administration", "Urban Planning"
        ]
    },
    "Veterinary": {
        "soft_skills": [
            "8. Communication skills", "7. Empathy", "6. Attention to detail", "5. Problem-solving",
            "4. Teamwork", "3. Adaptability", "2. Time management", "1. Customer service"
        ],
        "software_programs": [
            "Veterinary practice management software", "Microsoft Office Suite", "Medical imaging software", "Laboratory information systems"
        ],
        "study_fields": [
            "Veterinary Medicine", "Animal Science", "Zoology", "Biology", "Veterinary Technology"
        ]
    },
    "Warehousing": {
        "soft_skills": [
            "8. Teamwork", "7. Attention to detail", "6. Time management", "5. Organizational skills",
            "4. Communication skills", "3. Problem-solving", "2. Physical stamina", "1. Adaptability"
        ],
        "software_programs": [
            "Warehouse management systems (WMS)", "Microsoft Office Suite", "Inventory tracking software", "ERP systems"
        ],
        "study_fields": [
            "Supply Chain Management", "Logistics", "Business Administration", "Industrial Engineering", "Operations Management"
        ]
    }
}

# Cargar el archivo CSV desde el archivo zip
with zipfile.ZipFile('map_skills.zip', 'r') as zipf:
    with zipf.open('map_skills.csv') as f:
        df = pd.read_csv(f)

def show_home_page():
    st.title('EarnWise')
    
    # Añadir logotipo
    st.image('logo.png', width=150)

    st.write('''
    Welcome to EarnWise!

    The easiest, fastest and most transparent way to find salary information in your Sector/Industry.

    At this time, we only have information for the United States.

    Help us collect data from other countries by posting your information in the tab "Help Us Grow".
    ''')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("")
    with col2:
        find_insights = st.button('Find your salary insights')
        st.markdown("<h3 style='text-align: center;'>Or</h3>", unsafe_allow_html=True)
        predict_salary = st.button('Predict your salary')
    with col3:
        st.write("")

    return find_insights, predict_salary

def show_salary_insights():
    if st.button('Home'):
        st.session_state.page = "home"
    
    st.subheader('Filters')

    categories = sorted(df['Category'].unique().tolist())
    categories = ['All'] + categories
    category = st.selectbox('Category', categories)

    industries = df['Industry'].unique() if category == 'All' else df[df['Category'] == category]['Industry'].unique()
    industries = ['All'] + sorted(industries.tolist())
    industry = st.selectbox('Industry', industries)

    experiences = df['Experience Level'].unique() if industry == 'All' else df[(df['Category'] == category) & (df['Industry'] == industry)]['Experience Level'].unique()
    experiences = ['All'] + sorted(experiences.tolist())
    experience = st.selectbox('Experience Level', experiences)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Map', 'Salary Distribution', 'Salary Insights', 'Key Skills', 'Study Fields', 'Relevant Tools', 'Help Us Grow'])

    # Definir una escala de colores invertida
    custom_blues = [
        "rgb(222, 235, 247)",
        "rgb(198, 219, 239)",
        "rgb(158, 202, 225)",
        "rgb(107, 174, 214)",
        "rgb(66, 146, 198)",
        "rgb(33, 113, 181)",
        "rgb(8, 81, 156)",
        "rgb(8, 48, 107)"
    ]

    def update_map(category, industry, experience):
        filtered_df = df.copy()
        if category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == category]
        if industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == industry]
        if experience != 'All':
            filtered_df = filtered_df[filtered_df['Experience Level'] == experience]

        state_salary = filtered_df.groupby('State').agg(
            Medium_Salary=('Medium Salary', 'mean')
        ).reset_index()

        fig = px.choropleth(state_salary,
                            locations='State',
                            locationmode='USA-states',
                            color='Medium_Salary',
                            color_continuous_scale=custom_blues,
                            scope='usa',
                            labels={'Medium_Salary': 'Medium Salary'},
                            hover_data={'State': True, 'Medium_Salary': True})

        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>Medium Salary: $%{z:,.2f}'
        )

        fig.update_layout(title='Medium Salary by State', geo=dict(scope='usa'))
        st.plotly_chart(fig)

    def plot_wordcloud(category):
        if category != 'All':
            if category in industry_info:
                soft_skills = industry_info[category]['soft_skills']
                freq_dict = {skill.split('. ')[1]: int(skill.split('. ')[0]) for skill in soft_skills}
                
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate_from_frequencies(freq_dict)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Top Soft Skills in {category} Category')
                st.pyplot(plt)
            else:
                st.write(f'No soft skills data available for the selected category: {category}')
        else:
            st.write('Select a Category to Display Word Cloud')

    def plot_salary_distribution(category, industry):
        filtered_df = df.copy()
        if category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == category]
        if industry != 'All':
            filtered_df = filtered_df[filtered_df['Industry'] == industry]

        grouped_df = filtered_df.groupby('Experience Level', as_index=False)['Medium Salary'].mean()
        grouped_df = grouped_df.sort_values(by='Medium Salary')

        fig = px.bar(grouped_df, x='Experience Level', y='Medium Salary', color='Medium Salary',
                     color_continuous_scale=custom_blues,
                     title='Salary Distribution by Experience Level')
        fig.update_traces(texttemplate='$%{y:,.2f}', textposition='outside')

        fig.update_layout(yaxis_tickformat='$,.2f')

        st.plotly_chart(fig)

    def plot_salary_insights(category):
        filtered_df = df.copy()
        if category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == category]

        fig = px.box(filtered_df, x='Category', y='Medium Salary',
                     title='Salary Insights by Category',
                     color_discrete_sequence=['#08306b'])  # Usar un azul muy oscuro

        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Medium Salary: $%{y:,.2f}<extra></extra>'
        )

        st.plotly_chart(fig)

    def show_list(category, column):
        if category != 'All':
            filtered_df = df[df['Category'] == category]
            items = filtered_df[column].dropna().str.split(',').explode().unique()
            if items.size > 0:
                st.write(f"{column} in {category} Category:")
                for item in items:
                    st.write(f"- {item.strip()}")
            else:
                st.write(f'No data available for the selected category and column {column}')
        else:
            st.write(f'Select a Category to Display {column}')

    with tab1:
        update_map(category, industry, experience)

    with tab2:
        plot_salary_distribution(category, industry)

    with tab3:
        plot_salary_insights(category)

    with tab4:
        plot_wordcloud(category)

    with tab5:
        show_list(category, 'Study Fields')

    with tab6:
        show_list(category, 'Software Programs')

    with tab7:
        st.header('Help Us Grow')
        with st.form(key='help_us_grow_form'):
            country = st.text_input('Country:')
            state = st.text_input('State:')
            category_input = st.text_input('Category:')
            industry_input = st.text_input('Industry:')
            experience_level = st.text_input('Experience Level:')
            salary = st.text_input('Salary:')

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.write('Thank you for your submission!')

# Integrar la funcionalidad de predicción de salarios

# Función para cargar los datos
@st.cache_data
def load_data(zip_path, csv_name):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            with zipf.open(csv_name) as f:
                df = pd.read_csv(f)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df[["formatted_experience_level", "group_industry", "category", "state_formatted"]] = df[["formatted_experience_level", "group_industry", "category", "state_formatted"]].astype("string")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Función para preprocesar los datos
@st.cache_data
def preprocess_data(df):
    df_dummies = pd.get_dummies(df, columns=["formatted_experience_level", "group_industry", "category", "state_formatted"])
    x = df_dummies.drop(columns=['medium_salary', 'job_id', 'company_id', 'state', 'city', 'industry_id', 'job_title', 'description', 'location'])
    y = df_dummies['medium_salary']
    return x, y

# Entrenar el modelo (cargar el modelo entrenado previamente si existe)
@st.cache_resource
def train_model(x_train, y_train):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.09,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'seed': 42
    }
    num_rounds = 100  # Reducir el número de rondas de entrenamiento
    model = xgb.train(params, dtrain, num_rounds)
    return model

# Cargar y preprocesar los datos
df_prediction = load_data('df_clean_final.zip', 'df_clean_final - copia2.csv')
if df_prediction.empty:
    st.stop()

x, y = preprocess_data(df_prediction)

# Dividir los datos en conjuntos de entrenamiento y prueba (utilizar una muestra si es necesario)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_sample = x_train.sample(frac=0.1, random_state=42)  # Utilizar solo una muestra de los datos
y_train_sample = y_train.loc[x_train_sample.index]

model = train_model(x_train_sample, y_train_sample)

# Predecir y calcular métricas de desempeño
dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Función para predecir salarios por estado
def predict_state_salaries(experience_level, industry, category, min_salary, max_salary):
    predictions = []
    for state in df_prediction['state_formatted'].unique():
        employee_count = df_prediction.loc[df_prediction['state_formatted'] == state, 'employee_count'].values[0]
        input_data = pd.DataFrame({
            'formatted_experience_level': [experience_level],
            'group_industry': [industry],
            'category': [category],
            'state_formatted': [state],
            'minimum_salary': [min_salary],
            'maximum_salary': [max_salary],
            'employee_count': [employee_count]
        })
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=x.columns, fill_value=0)
        dinput = xgb.DMatrix(input_data)
        prediction = model.predict(dinput)
        predictions.append((state, prediction[0]))
    return predictions

def show_salary_prediction():
    if st.button('Home'):
        st.session_state.page = "home"

    st.title("Salary Prediction Dashboard")

    experience_level = st.selectbox("Experience Level:", df_prediction['formatted_experience_level'].unique())
    industry = st.selectbox("Industry Group:", df_prediction['group_industry'].unique())
    category = st.selectbox("Category:", df_prediction['category'].unique())
    state = st.selectbox("State:", df_prediction['state_formatted'].unique())
    min_salary = st.number_input("Minimum Salary:", value=0)
    max_salary = st.number_input("Maximum Salary:", value=0)

    if st.button('Predict'):
        employee_count = df_prediction.loc[df_prediction['state_formatted'] == state, 'employee_count'].values[0]
        
        input_data = pd.DataFrame({
            'formatted_experience_level': [experience_level],
            'group_industry': [industry],
            'category': [category],
            'state_formatted': [state],
            'minimum_salary': [min_salary],
            'maximum_salary': [max_salary],
            'employee_count': [employee_count]
        })
        
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=x.columns, fill_value=0)
        dinput = xgb.DMatrix(input_data)
        prediction = model.predict(dinput)[0]
        
        state_predictions = predict_state_salaries(experience_level, industry, category, min_salary, max_salary)
        prediction_df = pd.DataFrame(state_predictions, columns=['state', 'predicted_salary'])
        
        custom_blues = [
            "rgb(222, 235, 247)",
            "rgb(198, 219, 239)",
            "rgb(158, 202, 225)",
            "rgb(107, 174, 214)",
            "rgb(66, 146, 198)",
            "rgb(33, 113, 181)",
            "rgb(8, 81, 156)",
            "rgb(8, 48, 107)"
        ]
        
        fig = px.choropleth(
            prediction_df,
            locations='state',
            locationmode="USA-states",
            color='predicted_salary',
            scope="usa",
            color_continuous_scale=custom_blues,
            labels={'predicted_salary': 'Predicted Salary'}
        )
        
        st.markdown(f"<h3 style='text-align: center; color: green;'>Predicted Salary for {state}: ${prediction:,.2f}</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    find_insights, predict_salary = show_home_page()
    if find_insights:
        st.session_state.page = 'insights'
    elif predict_salary:
        st.session_state.page = 'prediction'
elif st.session_state.page == 'insights':
    show_salary_insights()
elif st.session_state.page == 'prediction':
    show_salary_prediction()
