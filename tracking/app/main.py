import os
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, List, Optional
import json

class StudentTracker:
    """Student activity tracking and analytics service"""
    
    def __init__(self):
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        self.client = MongoClient(mongodb_url)
        self.db = self.client.minecraft_edu
        
        # Collections
        self.sessions = self.db.student_sessions
        self.activities = self.db.student_activities  
        self.chat_logs = self.db.chat_interactions
        self.progress = self.db.learning_progress
        
        print("📊 Student Tracker initialized")
    
    def start_session(self, user_id: str, username: str, minecraft_version: str = None) -> str:
        """Start a new learning session"""
        session_data = {
            'user_id': user_id,
            'username': username,
            'start_time': datetime.now(),
            'end_time': None,
            'duration_minutes': None,
            'minecraft_version': minecraft_version,
            'activities_count': 0,
            'chat_messages_count': 0,
            'learning_objectives_met': [],
            'status': 'active'
        }
        
        # End any existing active sessions
        self.sessions.update_many(
            {'user_id': user_id, 'status': 'active'},
            {'$set': {'status': 'interrupted', 'end_time': datetime.now()}}
        )
        
        # Insert new session
        result = self.sessions.insert_one(session_data)
        session_id = str(result.inserted_id)
        
        print(f"🎯 Session started: {username} ({user_id}) - {session_id}")
        return session_id
    
    def end_session(self, user_id: str) -> Optional[Dict]:
        """End the active learning session"""
        session = self.sessions.find_one({
            'user_id': user_id,
            'status': 'active'
        })
        
        if not session:
            print(f"⚠️ No active session found for {user_id}")
            return None
            
        end_time = datetime.now()
        start_time = session['start_time']
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Update session
        update_data = {
            'end_time': end_time,
            'duration_minutes': round(duration, 2),
            'status': 'completed'
        }
        
        self.sessions.update_one(
            {'_id': session['_id']},
            {'$set': update_data}
        )
        
        session_summary = {
            'session_id': str(session['_id']),
            'duration_minutes': round(duration, 2),
            'activities_count': session['activities_count'],
            'chat_messages_count': session['chat_messages_count'],
            'learning_objectives_met': session['learning_objectives_met']
        }
        
        print(f"🏁 Session ended: {user_id} - {duration:.1f} minutes")
        return session_summary
    
    def log_activity(self, user_id: str, activity_type: str, details: Dict) -> str:
        """Log a student activity"""
        activity_data = {
            'user_id': user_id,
            'activity_type': activity_type,
            'timestamp': datetime.now(),
            'details': details,
            'minecraft_coordinates': details.get('coordinates'),
            'blocks_used': details.get('blocks', []),
            'commands_executed': details.get('commands', [])
        }
        
        # Insert activity
        result = self.activities.insert_one(activity_data)
        activity_id = str(result.inserted_id)
        
        # Update session activity count
        self.sessions.update_one(
            {'user_id': user_id, 'status': 'active'},
            {'$inc': {'activities_count': 1}}
        )
        
        # Check for learning objectives met
        self._check_learning_objectives(user_id, activity_type, details)
        
        print(f"🎮 Activity logged: {user_id} - {activity_type}")
        return activity_id
    
    def log_chat_interaction(self, user_id: str, user_message: str, ai_response: str, 
                           subject: str = None, topic: str = None) -> str:
        """Log AI chat interaction"""
        chat_data = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'user_message': user_message,
            'ai_response': ai_response,
            'subject': subject,
            'topic': topic,
            'message_length': len(user_message),
            'response_length': len(ai_response),
            'sentiment': self._analyze_sentiment(user_message)
        }
        
        # Insert chat log
        result = self.chat_logs.insert_one(chat_data)
        chat_id = str(result.inserted_id)
        
        # Update session chat count
        self.sessions.update_one(
            {'user_id': user_id, 'status': 'active'},
            {'$inc': {'chat_messages_count': 1}}
        )
        
        print(f"💬 Chat logged: {user_id} - {len(user_message)} chars")
        return chat_id
    
    def get_student_analytics(self, user_id: str, days: int = 7) -> Dict:
        """Get comprehensive student analytics"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Session analytics
        sessions = list(self.sessions.find({
            'user_id': user_id,
            'start_time': {'$gte': cutoff_date}
        }))
        
        # Activity analytics
        activities = list(self.activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        # Chat analytics
        chats = list(self.chat_logs.find({
            'user_id': user_id,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        # Calculate metrics
        total_time = sum(s.get('duration_minutes', 0) for s in sessions if s.get('duration_minutes'))
        activity_types = list(set(a['activity_type'] for a in activities))
        subjects_discussed = list(set(c.get('subject') for c in chats if c.get('subject')))
        
        # Learning objectives met
        all_objectives = []
        for session in sessions:
            all_objectives.extend(session.get('learning_objectives_met', []))
        unique_objectives = list(set(all_objectives))
        
        analytics = {
            'user_id': user_id,
            'period_days': days,
            'summary': {
                'total_sessions': len(sessions),
                'total_time_minutes': round(total_time, 2),
                'average_session_minutes': round(total_time / len(sessions), 2) if sessions else 0,
                'total_activities': len(activities),
                'total_chat_messages': len(chats)
            },
            'engagement': {
                'activity_types': activity_types,
                'subjects_discussed': subjects_discussed,
                'learning_objectives_met': unique_objectives,
                'most_active_day': self._get_most_active_day(activities)
            },
            'progress': {
                'completion_rate': self._calculate_completion_rate(user_id),
                'difficulty_progression': self._get_difficulty_progression(user_id),
                'strengths': self._identify_strengths(activities, chats),
                'areas_for_improvement': self._identify_improvement_areas(activities, chats)
            }
        }
        
        return analytics
    
    def _check_learning_objectives(self, user_id: str, activity_type: str, details: Dict):
        """Check if activity meets learning objectives"""
        objectives_met = []
        
        if activity_type == 'building' and details.get('shape') == 'square':
            objectives_met.append('spatial_reasoning')
            objectives_met.append('geometry_basics')
        elif activity_type == 'redstone_circuit':
            objectives_met.append('basic_circuits')
            objectives_met.append('logical_thinking')
        elif activity_type == 'measurement':
            objectives_met.append('mathematical_calculation')
        
        if objectives_met:
            self.sessions.update_one(
                {'user_id': user_id, 'status': 'active'},
                {'$addToSet': {'learning_objectives_met': {'$each': objectives_met}}}
            )
    
    def _analyze_sentiment(self, message: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'awesome', 'cool', 'fun', 'like', 'love']
        negative_words = ['bad', 'hate', 'difficult', 'hard', 'confused', 'boring']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_most_active_day(self, activities: List[Dict]) -> str:
        """Find the most active day of the week"""
        if not activities:
            return None
        
        day_counts = {}
        for activity in activities:
            day = activity['timestamp'].strftime('%A')
            day_counts[day] = day_counts.get(day, 0) + 1
        
        return max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None
    
    def _calculate_completion_rate(self, user_id: str) -> float:
        """Calculate learning objective completion rate"""
        # Simplified - in real implementation would track against curriculum
        total_objectives = 20  # Example total
        completed = len(self.sessions.distinct('learning_objectives_met', {'user_id': user_id}))
        return min(completed / total_objectives * 100, 100)
    
    def _get_difficulty_progression(self, user_id: str) -> str:
        """Analyze difficulty progression"""
        # Simplified analysis
        activities = list(self.activities.find(
            {'user_id': user_id},
            {'details.difficulty': 1, 'timestamp': 1}
        ).sort('timestamp', 1))
        
        if len(activities) < 2:
            return 'insufficient_data'
        
        # Check if moving from beginner to intermediate/advanced
        recent_activities = activities[-5:]  # Last 5 activities
        difficulties = [a.get('details', {}).get('difficulty') for a in recent_activities]
        
        if 'advanced' in difficulties:
            return 'advanced'
        elif 'intermediate' in difficulties:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _identify_strengths(self, activities: List[Dict], chats: List[Dict]) -> List[str]:
        """Identify student strengths based on activity patterns"""
        strengths = []
        
        # Analyze activity types
        activity_counts = {}
        for activity in activities:
            activity_type = activity['activity_type']
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # High activity in specific areas indicates strength
        if activity_counts.get('building', 0) > 10:
            strengths.append('spatial_construction')
        if activity_counts.get('redstone_circuit', 0) > 5:
            strengths.append('logical_reasoning')
        if activity_counts.get('measurement', 0) > 5:
            strengths.append('mathematical_thinking')
        
        # Analyze chat subjects
        subjects = [c.get('subject') for c in chats if c.get('subject')]
        subject_counts = {}
        for subject in subjects:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        if subject_counts.get('mathematics', 0) > 10:
            strengths.append('mathematics_engagement')
        if subject_counts.get('physics', 0) > 5:
            strengths.append('physics_curiosity')
        
        return strengths[:3]  # Top 3 strengths
    
    def _identify_improvement_areas(self, activities: List[Dict], chats: List[Dict]) -> List[str]:
        """Identify areas needing improvement"""
        improvements = []
        
        # Low activity in certain areas
        activity_counts = {}
        for activity in activities:
            activity_type = activity['activity_type']
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        if activity_counts.get('measurement', 0) < 2:
            improvements.append('mathematical_measurement')
        if activity_counts.get('planning', 0) < 2:
            improvements.append('project_planning')
        
        # Analyze chat sentiment for frustration patterns
        negative_chats = [c for c in chats if c.get('sentiment') == 'negative']
        if len(negative_chats) > len(chats) * 0.3:  # More than 30% negative
            improvements.append('confidence_building')
        
        return improvements[:3]  # Top 3 improvement areas

# Factory function for easy import
def create_tracker() -> StudentTracker:
    """Create and return a StudentTracker instance"""
    return StudentTracker()

if __name__ == "__main__":
    # Demo usage
    tracker = create_tracker()
    
    # Test basic functionality
    session_id = tracker.start_session("demo_user", "Demo Student")
    activity_id = tracker.log_activity("demo_user", "building", {
        "shape": "square",
        "size": "5x5",
        "coordinates": {"x": 100, "y": 64, "z": 200},
        "blocks": ["stone", "glass"],
        "difficulty": "beginner"
    })
    
    chat_id = tracker.log_chat_interaction(
        "demo_user",
        "How do I calculate the area of a square?",
        "To calculate the area of a square, multiply the length by the width. For a 5x5 square, the area is 5 × 5 = 25 blocks!",
        "mathematics",
        "geometry"
    )
    
    summary = tracker.end_session("demo_user")
    analytics = tracker.get_student_analytics("demo_user")
    
    print("\n📈 Demo Analytics:")
    print(json.dumps(analytics, indent=2, default=str))