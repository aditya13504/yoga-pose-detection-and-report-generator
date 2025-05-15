import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from fpdf import FPDF
import time
import joblib
from pathlib import Path

# --- Report Configuration ---
REPORT_OUTPUT_DIR = "reports"
USER_DATA_FILE = "user_data.json"
SESSION_DATA_DIR = "session_data"

class YogaReportGenerator:
    def __init__(self):
        # Create necessary directories
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(SESSION_DATA_DIR, exist_ok=True)
        
        # Initialize or load user data
        self.user_data = self._load_user_data()
        
    def _load_user_data(self):
        """Load user profile data from JSON file or create if not exists"""
        if os.path.exists(USER_DATA_FILE):
            try:
                with open(USER_DATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user data: {e}")
                return self._create_default_user_data()
        else:
            return self._create_default_user_data()
    
    def _create_default_user_data(self):
        """Create default user data structure"""
        return {
            "personal_info": {
                "name": "User",
                "age": None,
                "height": None,
                "weight": None,
                "fitness_level": None  # beginner, intermediate, advanced
            },
            "medical_history": {
                "conditions": [],
                "injuries": [],
                "limitations": []
            },
            "yoga_experience": {
                "years_practicing": None,
                "frequency_per_week": None,
                "favorite_poses": [],
                "challenging_poses": []
            }
        }
    
    def collect_user_data(self, interactive=True):
        """Collect or update user data"""
        if interactive:
            print("\n--- User Profile Setup ---")
            
            # Personal info
            self.user_data["personal_info"]["name"] = input("Name: ").strip() or self.user_data["personal_info"]["name"]
            
            try:
                age = input("Age: ").strip()
                self.user_data["personal_info"]["age"] = int(age) if age else self.user_data["personal_info"]["age"]
            except ValueError:
                print("Invalid age input. Using previous value.")
            
            try:
                height = input("Height (cm): ").strip()
                self.user_data["personal_info"]["height"] = float(height) if height else self.user_data["personal_info"]["height"]
            except ValueError:
                print("Invalid height input. Using previous value.")
            
            try:
                weight = input("Weight (kg): ").strip()
                self.user_data["personal_info"]["weight"] = float(weight) if weight else self.user_data["personal_info"]["weight"]
            except ValueError:
                print("Invalid weight input. Using previous value.")
            
            # Fitness level
            print("\nFitness level:")
            print("1. Beginner")
            print("2. Intermediate")
            print("3. Advanced")
            fitness_choice = input("Select (1-3): ").strip()
            if fitness_choice:
                fitness_mapping = {"1": "beginner", "2": "intermediate", "3": "advanced"}
                self.user_data["personal_info"]["fitness_level"] = fitness_mapping.get(fitness_choice, self.user_data["personal_info"]["fitness_level"])
            
            # Medical history
            print("\n--- Medical History ---")
            print("Enter any medical conditions (comma-separated, leave blank to keep current):")
            conditions = input().strip()
            if conditions:
                self.user_data["medical_history"]["conditions"] = [c.strip() for c in conditions.split(",")]
            
            print("Enter any injuries (comma-separated, leave blank to keep current):")
            injuries = input().strip()
            if injuries:
                self.user_data["medical_history"]["injuries"] = [i.strip() for i in injuries.split(",")]
            
            print("Enter any physical limitations (comma-separated, leave blank to keep current):")
            limitations = input().strip()
            if limitations:
                self.user_data["medical_history"]["limitations"] = [l.strip() for l in limitations.split(",")]
            
            # Yoga experience
            print("\n--- Yoga Experience ---")
            try:
                years = input("Years practicing yoga: ").strip()
                self.user_data["yoga_experience"]["years_practicing"] = float(years) if years else self.user_data["yoga_experience"]["years_practicing"]
            except ValueError:
                print("Invalid input. Using previous value.")
            
            try:
                freq = input("Practice frequency (times per week): ").strip()
                self.user_data["yoga_experience"]["frequency_per_week"] = int(freq) if freq else self.user_data["yoga_experience"]["frequency_per_week"]
            except ValueError:
                print("Invalid input. Using previous value.")
            
            print("Favorite poses (comma-separated, leave blank to keep current):")
            fav_poses = input().strip()
            if fav_poses:
                self.user_data["yoga_experience"]["favorite_poses"] = [p.strip() for p in fav_poses.split(",")]
            
            print("Challenging poses (comma-separated, leave blank to keep current):")
            chall_poses = input().strip()
            if chall_poses:
                self.user_data["yoga_experience"]["challenging_poses"] = [p.strip() for p in chall_poses.split(",")]
            
        # Save updated user data
        self._save_user_data()
        print("\nUser profile updated successfully.")
    
    def _save_user_data(self):
        """Save user data to JSON file"""
        try:
            with open(USER_DATA_FILE, 'w') as f:
                json.dump(self.user_data, f, indent=4)
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def save_session_data(self, session_data):
        """Save a single yoga session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = os.path.join(SESSION_DATA_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=4)
            print(f"Session data saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving session data: {e}")
            return None
    
    def analyze_session(self, session_data):
        """Analyze a single yoga session"""
        results = {
            "timestamp": session_data.get("timestamp", "Unknown"),
            "duration": session_data.get("duration", 0),
            "pose_type": session_data.get("pose_type", "Unknown"),
            "attempts": session_data.get("attempts", 0),
            "successful_poses": len(session_data.get("correct_poses", [])),
            "accuracy": 0,
            "avg_confidence": 0,
            "common_issues": [],
            "improvement_areas": []
        }
        
        # Calculate accuracy
        if results["attempts"] > 0:
            results["accuracy"] = (results["successful_poses"] / results["attempts"]) * 100
        
        # Calculate average confidence
        confidences = [pose.get("confidence", 0) for pose in session_data.get("correct_poses", [])]
        if confidences:
            results["avg_confidence"] = sum(confidences) / len(confidences)
        
        # Analyze feedback messages to identify common issues
        feedback_messages = session_data.get("feedback_history", [])
        issue_count = {}
        for msg in feedback_messages:
            # Skip generic messages
            if msg in ["Adjust pose.", "No pose detected."]:
                continue
            
            issue_count[msg] = issue_count.get(msg, 0) + 1
        
        # Get the top 3 most common issues
        sorted_issues = sorted(issue_count.items(), key=lambda x: x[1], reverse=True)
        results["common_issues"] = [issue for issue, count in sorted_issues[:3]]
        
        # Generate improvement areas based on common issues
        results["improvement_areas"] = self._generate_improvement_areas(results["common_issues"])
        
        return results
    
    def _generate_improvement_areas(self, common_issues):
        """Generate specific improvement areas based on common issues"""
        improvement_areas = []
        
        for issue in common_issues:
            if "straighten your left" in issue.lower():
                improvement_areas.append("Practice single-leg balance exercises to strengthen your left leg")
            elif "straighten your right" in issue.lower():
                improvement_areas.append("Practice single-leg balance exercises to strengthen your right leg")
            elif "foot higher" in issue.lower():
                improvement_areas.append("Work on hip flexibility to better position your foot")
            elif "hands closer together" in issue.lower():
                improvement_areas.append("Practice shoulder and arm alignment in mountain pose first")
            elif "make sure both hands are visible" in issue.lower():
                improvement_areas.append("Adjust your position in front of the camera for better visibility")
            else:
                improvement_areas.append("Review proper Tree Pose alignment in a yoga class or with an instructor")
        
        # Add general recommendations
        if self.user_data["personal_info"]["fitness_level"] == "beginner":
            improvement_areas.append("Try practicing against a wall for better balance")
        
        # Check for medical concerns
        if any("back" in cond.lower() for cond in self.user_data["medical_history"]["conditions"]):
            improvement_areas.append("Consult with your doctor about back-safe modifications for this pose")
        
        if any("knee" in inj.lower() for inj in self.user_data["medical_history"]["injuries"]):
            improvement_areas.append("Be gentle with knee pressure and consider lower foot placement")
        
        return improvement_areas
    
    def analyze_all_sessions(self, pose_type="tree"):
        """Analyze all sessions of a specific pose type"""
        sessions = []
        
        # Load all session files
        for filename in os.listdir(SESSION_DATA_DIR):
            if not filename.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(SESSION_DATA_DIR, filename), 'r') as f:
                    session_data = json.load(f)
                    if session_data.get("pose_type", "") == pose_type:
                        sessions.append(session_data)
            except Exception as e:
                print(f"Error loading session file {filename}: {e}")
        
        if not sessions:
            return {
                "pose_type": pose_type,
                "total_sessions": 0,
                "message": "No session data found for this pose type."
            }
        
        # Sort sessions by timestamp
        sessions.sort(key=lambda x: x.get("timestamp", ""))
        
        # Analyze progress over time
        results = {
            "pose_type": pose_type,
            "total_sessions": len(sessions),
            "first_session_date": sessions[0].get("timestamp", "Unknown"),
            "latest_session_date": sessions[-1].get("timestamp", "Unknown"),
            "accuracy_trend": [],
            "confidence_trend": [],
            "total_attempts": sum(s.get("attempts", 0) for s in sessions),
            "total_successful": sum(len(s.get("correct_poses", [])) for s in sessions),
            "average_session_duration": sum(s.get("duration", 0) for s in sessions) / len(sessions),
            "common_issues": {},
            "health_concerns": [],
            "recommendations": []
        }
        
        # Calculate overall accuracy
        if results["total_attempts"] > 0:
            results["overall_accuracy"] = (results["total_successful"] / results["total_attempts"]) * 100
        else:
            results["overall_accuracy"] = 0
        
        # Track accuracy and confidence over time
        for session in sessions:
            attempts = session.get("attempts", 0)
            successful = len(session.get("correct_poses", []))
            accuracy = (successful / attempts) * 100 if attempts > 0 else 0
            
            confidences = [pose.get("confidence", 0) for pose in session.get("correct_poses", [])]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            results["accuracy_trend"].append({
                "date": session.get("timestamp", "Unknown"),
                "accuracy": accuracy
            })
            
            results["confidence_trend"].append({
                "date": session.get("timestamp", "Unknown"),
                "confidence": avg_confidence
            })
        
        # Compile all feedback to identify persistent issues
        all_feedback = []
        for session in sessions:
            all_feedback.extend(session.get("feedback_history", []))
        
        # Count occurrences of each feedback message
        for msg in all_feedback:
            if msg in ["Adjust pose.", "No pose detected."]:
                continue
            results["common_issues"][msg] = results["common_issues"].get(msg, 0) + 1
        
        # Sort issues by frequency
        results["common_issues"] = dict(sorted(
            results["common_issues"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Identify potential health concerns based on persistent issues
        top_issues = list(results["common_issues"].keys())[:3]
        health_concerns = self._identify_health_concerns(top_issues)
        results["health_concerns"] = health_concerns
        
        # Generate personalized recommendations
        results["recommendations"] = self._generate_recommendations(results, health_concerns)
        
        return results
    
    def _identify_health_concerns(self, common_issues):
        """Identify potential health concerns based on persistent issues"""
        concerns = []
        
        leg_issues_count = sum(1 for issue in common_issues if "leg" in issue.lower() or "foot" in issue.lower())
        balance_issues_count = sum(1 for issue in common_issues if "balance" in issue.lower() or "straighten" in issue.lower())
        alignment_issues_count = sum(1 for issue in common_issues if "alignment" in issue.lower())
        
        # Check for medical conditions
        conditions = self.user_data["medical_history"]["conditions"]
        injuries = self.user_data["medical_history"]["injuries"]
        
        if leg_issues_count >= 2:
            concerns.append({
                "area": "Lower Body",
                "description": "Repeated issues with leg positioning may indicate potential balance or strength issues",
                "risk_level": "Medium" if any("knee" in inj.lower() for inj in injuries) else "Low"
            })
        
        if balance_issues_count >= 2:
            concerns.append({
                "area": "Balance",
                "description": "Consistent balance difficulties could suggest inner ear issues or core weakness",
                "risk_level": "Medium" if any("vertigo" in cond.lower() or "dizzy" in cond.lower() for cond in conditions) else "Low"
            })
        
        if alignment_issues_count >= 2:
            concerns.append({
                "area": "Posture",
                "description": "Frequent alignment problems may indicate postural imbalances that could cause strain over time",
                "risk_level": "Medium" if any("back" in cond.lower() for cond in conditions) else "Low"
            })
        
        # Add age-specific concerns
        age = self.user_data["personal_info"]["age"]
        if age and age > 65 and balance_issues_count > 0:
            concerns.append({
                "area": "Fall Risk",
                "description": "Balance issues at this age group increase risk of falls",
                "risk_level": "Medium"
            })
        
        return concerns
    
    def _generate_recommendations(self, results, health_concerns):
        """Generate personalized recommendations based on analysis and user profile"""
        recommendations = []
        
        # Basic recommendations based on accuracy
        if results["overall_accuracy"] < 30:
            recommendations.append({
                "category": "Fundamentals",
                "description": "Focus on mastering mountain pose first to build a strong foundation",
                "priority": "High"
            })
        elif results["overall_accuracy"] < 60:
            recommendations.append({
                "category": "Practice",
                "description": "Try practicing against a wall first for better stability",
                "priority": "Medium"
            })
        else:
            recommendations.append({
                "category": "Advancement",
                "description": "You're showing good progress! Try holding the pose for longer periods",
                "priority": "Low"
            })
        
        # Add recommendations based on fitness level
        fitness_level = self.user_data["personal_info"]["fitness_level"]
        if fitness_level == "beginner":
            recommendations.append({
                "category": "Strength",
                "description": "Include simple balance exercises in your daily routine to improve stability",
                "priority": "High"
            })
        elif fitness_level == "intermediate":
            recommendations.append({
                "category": "Flexibility",
                "description": "Add hip-opening poses to your practice to improve foot placement in tree pose",
                "priority": "Medium"
            })
        elif fitness_level == "advanced":
            recommendations.append({
                "category": "Refinement",
                "description": "Work on subtle alignment adjustments and breath control to perfect your tree pose",
                "priority": "Medium"
            })
        
        # Add recommendations based on identified health concerns
        for concern in health_concerns:
            if concern["area"] == "Lower Body":
                recommendations.append({
                    "category": "Strengthening",
                    "description": "Incorporate lower body strength exercises like gentle squats and calf raises",
                    "priority": "High" if concern["risk_level"] == "Medium" else "Medium"
                })
            elif concern["area"] == "Balance":
                recommendations.append({
                    "category": "Balance",
                    "description": "Practice simple balance exercises daily, gradually increasing difficulty",
                    "priority": "High" if concern["risk_level"] == "Medium" else "Medium"
                })
            elif concern["area"] == "Posture":
                recommendations.append({
                    "category": "Alignment",
                    "description": "Focus on proper spinal alignment in all poses, consider postural exercises",
                    "priority": "High" if concern["risk_level"] == "Medium" else "Medium"
                })
            elif concern["area"] == "Fall Risk":
                recommendations.append({
                    "category": "Safety",
                    "description": "Always practice near a wall or with a chair for support to prevent falls",
                    "priority": "High"
                })
        
        # Add medical history based recommendations
        if any("back" in cond.lower() for cond in self.user_data["medical_history"]["conditions"]):
            recommendations.append({
                "category": "Medical",
                "description": "Consult with your healthcare provider about back-safe modifications for tree pose",
                "priority": "High"
            })
        
        if any("knee" in inj.lower() for inj in self.user_data["medical_history"]["injuries"]):
            recommendations.append({
                "category": "Injury Management",
                "description": "Place your foot lower on your leg (calf instead of thigh) to reduce knee strain",
                "priority": "High"
            })
        
        return recommendations
    
    def generate_report_pdf(self, analysis_results):
        """Generate a PDF report with analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"yoga_report_{timestamp}.pdf"
        report_path = os.path.join(REPORT_OUTPUT_DIR, report_filename)
        
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Yoga Practice Analysis Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Add date
        pdf.set_font('Arial', '', 10)
        pdf.cell(190, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
        
        # Add user info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'User Profile', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        name = self.user_data["personal_info"]["name"]
        age = self.user_data["personal_info"]["age"] or "Not specified"
        fitness_level = self.user_data["personal_info"]["fitness_level"] or "Not specified"
        pdf.cell(190, 10, f'Name: {name}', 0, 1)
        pdf.cell(190, 10, f'Age: {age}', 0, 1)
        pdf.cell(190, 10, f'Fitness Level: {fitness_level}', 0, 1)
        
        # Add session summary
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Practice Summary', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(190, 10, f'Pose Type: {analysis_results["pose_type"].capitalize()} Pose', 0, 1)
        pdf.cell(190, 10, f'Total Sessions: {analysis_results["total_sessions"]}', 0, 1)
        pdf.cell(190, 10, f'Overall Accuracy: {analysis_results.get("overall_accuracy", 0):.1f}%', 0, 1)
        pdf.cell(190, 10, f'Total Successful Poses: {analysis_results["total_successful"]}', 0, 1)
        
        # Add performance trends section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Performance Analysis', 0, 1)
        
        # Save accuracy trend graph if data exists
        if analysis_results.get("accuracy_trend"):
            # Create graph for accuracy trend
            plt.figure(figsize=(10, 4))
            dates = [item["date"] for item in analysis_results["accuracy_trend"]]
            accuracies = [item["accuracy"] for item in analysis_results["accuracy_trend"]]
            
            # Format dates for better display
            short_dates = [d.split("_")[0] if "_" in d else d for d in dates]
            
            plt.plot(short_dates, accuracies, marker='o')
            plt.title('Pose Accuracy Over Time')
            plt.xlabel('Session Date')
            plt.ylabel('Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save graph
            graph_path = os.path.join(REPORT_OUTPUT_DIR, "accuracy_trend.png")
            plt.savefig(graph_path)
            plt.close()
            
            # Add graph to PDF
            pdf.ln(5)
            pdf.cell(190, 10, 'Accuracy Trend:', 0, 1)
            pdf.image(graph_path, x=10, y=None, w=180)
            pdf.ln(60)  # Space for the image
        
        # Add common issues section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Most Common Issues', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        if analysis_results.get("common_issues"):
            for i, (issue, count) in enumerate(list(analysis_results["common_issues"].items())[:5]):
                pdf.cell(190, 10, f'{i+1}. {issue} ({count} occurrences)', 0, 1)
        else:
            pdf.cell(190, 10, 'No common issues identified.', 0, 1)
        
        # Add health concerns section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Potential Health Considerations', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        if analysis_results.get("health_concerns"):
            for concern in analysis_results["health_concerns"]:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(190, 10, f'{concern["area"]} (Risk Level: {concern["risk_level"]})', 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(190, 10, concern["description"])
        else:
            pdf.cell(190, 10, 'No specific health concerns identified.', 0, 1)
        
        # Add recommendations section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Personalized Recommendations', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        if analysis_results.get("recommendations"):
            for i, rec in enumerate(analysis_results["recommendations"]):
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(190, 10, f'{i+1}. {rec["category"]} (Priority: {rec["priority"]})', 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(190, 10, rec["description"])
                pdf.ln(2)
        else:
            pdf.cell(190, 10, 'No specific recommendations available.', 0, 1)
        
        # Add disclaimer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.multi_cell(190, 10, 'Disclaimer: This report is generated based on automated analysis and should not replace professional medical or fitness advice. Always consult with qualified professionals for personalized guidance.')
        
        # Save PDF
        try:
            pdf.output(report_path)
            print(f"Report generated and saved to {report_path}")
            return report_path
        except Exception as e:
            print(f"Error saving PDF report: {e}")
            return None

# For testing
if __name__ == "__main__":
    report_gen = YogaReportGenerator()
    
    # Example: Collect user data
    # report_gen.collect_user_data()
    
    # Example: Create dummy session data for testing
    # test_session = {
    #     "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    #     "duration": 120,
    #     "pose_type": "tree",
    #     "attempts": 10,
    #     "correct_poses": [
    #         {"timestamp": "20220101_120000", "confidence": 0.85},
    #         {"timestamp": "20220101_120030", "confidence": 0.92}
    #     ],
    #     "feedback_history": [
    #         "Adjust pose.",
    #         "Straighten your left standing leg.",
    #         "Place your right foot higher on your left inner thigh.",
    #         "Bring your hands closer together at your chest."
    #     ]
    # }
    # report_gen.save_session_data(test_session)
    
    # Example: Analyze all sessions and generate report
    # analysis = report_gen.analyze_all_sessions("tree")
    # report_gen.generate_report_pdf(analysis) 