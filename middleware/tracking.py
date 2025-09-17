from pymongo import MongoClient
from datetime import datetime
import os
from typing import Dict, Any, Optional

class StudentTracker:
    """Service for tracking student activities and learning analytics"""
    
    def __init__(self):
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        self.client = MongoClient(mongodb_url)
        self.db = self.client.minecraft_edu
        
        # Collections
        self.sessions = self.db.student_sessions
        self.activities = self.db.student_activities
        self.chat_logs = self.db.chat_interactions
        
        print("📊 Student Tracker initialized")
    
    async def log_session_start(self, user_id: str, username: str):
        """Log when a student starts a learning session"""
        session_data = {
            'user_id': user_id,
            'username': username,
            'start_time': datetime.now(),
            'end_time': None,
            'duration_minutes': None,
            'activities_count': 0,
            'chat_messages_count': 0,
            'status': 'active'
        }
        
        # Upsert session (update if exists, insert if not)
        self.sessions.replace_one(
            {'user_id': user_id, 'status': 'active'},
            session_data,
            upsert=True
        )
        
        print(f"📝 Session started for {username} ({user_id})")
    
    async def log_session_end(self, user_id: str):
        """Log when a student ends a learning session"""
        # Find active session
        session = self.sessions.find_one({
            'user_id': user_id, 
            'status': 'active'
        })
        
        if session:
            end_time = datetime.now()
            start_time = session['start_time']
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Update session with end data
            self.sessions.update_one(
                {'_id': session['_id']},
                {
                    '$set': {
                        'end_time': end_time,
                        'duration_minutes': duration,
                        'status': 'completed'
                    }
                }
            )
            
            print(f"🏁 Session ended for {user_id}, duration: {duration:.1f} minutes")
    
    async def log_activity(self, user_id: str, activity_type: str, details: Dict[str, Any]):
        """Log a student activity in Minecraft"""
        activity_data = {
            'user_id': user_id,
            'activity_type': activity_type,
            'timestamp': datetime.now(),
            'details': details
        }
        
        # Insert activity log
        self.activities.insert_one(activity_data)
        
        # Update session activity count
        self.sessions.update_one(
            {'user_id': user_id, 'status': 'active'},
            {'$inc': {'activities_count': 1}}
        )
        
        print(f"🎯 Activity logged: {user_id} - {activity_type}")
    
    async def log_chat_interaction(self, user_id: str, user_message: str, ai_response: str):
        """Log a chat interaction between student and AI"""
        chat_data = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'user_message': user_message,
            'ai_response': ai_response,
            'message_length': len(user_message),
            'response_length': len(ai_response)
        }
        
        # Insert chat log
        self.chat_logs.insert_one(chat_data)
        
        # Update session chat count
        self.sessions.update_one(
            {'user_id': user_id, 'status': 'active'},
            {'$inc': {'chat_messages_count': 1}}
        )
        
        print(f"💬 Chat logged: {user_id} - {len(user_message)} chars")
    
    def get_student_stats(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get learning statistics for a student"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Session stats
        sessions = list(self.sessions.find({
            'user_id': user_id,
            'start_time': {'$gte': cutoff_date}
        }))
        
        # Activity stats
        activities = list(self.activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        # Chat stats
        chats = list(self.chat_logs.find({
            'user_id': user_id,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        return {
            'user_id': user_id,
            'period_days': days,
            'total_sessions': len(sessions),
            'total_time_minutes': sum(s.get('duration_minutes', 0) for s in sessions if s.get('duration_minutes')),
            'total_activities': len(activities),
            'total_chat_messages': len(chats),
            'activity_types': list(set(a['activity_type'] for a in activities)),
            'last_session': sessions[0]['start_time'] if sessions else None
        }