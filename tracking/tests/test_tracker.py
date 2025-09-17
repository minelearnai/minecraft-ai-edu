import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from app.main import StudentTracker, create_tracker

@pytest.fixture
def mock_mongo():
    """Mock MongoDB client and collections"""
    with patch('app.main.MongoClient') as mock_client:
        mock_db = Mock()
        mock_client.return_value.minecraft_edu = mock_db
        
        # Mock collections
        mock_db.student_sessions = Mock()
        mock_db.student_activities = Mock()
        mock_db.chat_interactions = Mock()
        mock_db.learning_progress = Mock()
        
        yield mock_db

def test_create_tracker():
    """Test tracker creation"""
    with patch('app.main.MongoClient'):
        tracker = create_tracker()
        assert isinstance(tracker, StudentTracker)

def test_start_session(mock_mongo):
    """Test starting a learning session"""
    tracker = StudentTracker()
    
    # Mock insert result
    mock_result = Mock()
    mock_result.inserted_id = '507f1f77bcf86cd799439011'
    mock_mongo.student_sessions.insert_one.return_value = mock_result
    
    session_id = tracker.start_session('test_user', 'Test Student')
    
    assert session_id == '507f1f77bcf86cd799439011'
    mock_mongo.student_sessions.insert_one.assert_called_once()
    
    # Check that previous active sessions were ended
    mock_mongo.student_sessions.update_many.assert_called_once()

def test_log_activity(mock_mongo):
    """Test logging student activity"""
    tracker = StudentTracker()
    
    # Mock insert result
    mock_result = Mock()
    mock_result.inserted_id = '507f1f77bcf86cd799439012'
    mock_mongo.student_activities.insert_one.return_value = mock_result
    
    activity_details = {
        'shape': 'square',
        'size': '5x5',
        'coordinates': {'x': 100, 'y': 64, 'z': 200}
    }
    
    activity_id = tracker.log_activity('test_user', 'building', activity_details)
    
    assert activity_id == '507f1f77bcf86cd799439012'
    mock_mongo.student_activities.insert_one.assert_called_once()
    
    # Check that session activity count was incremented
    mock_mongo.student_sessions.update_one.assert_called_once()

def test_log_chat_interaction(mock_mongo):
    """Test logging AI chat interaction"""
    tracker = StudentTracker()
    
    # Mock insert result
    mock_result = Mock()
    mock_result.inserted_id = '507f1f77bcf86cd799439013'
    mock_mongo.chat_interactions.insert_one.return_value = mock_result
    
    chat_id = tracker.log_chat_interaction(
        'test_user',
        'How do I build a square?',
        'To build a square, place blocks in equal rows and columns.',
        'mathematics',
        'geometry'
    )
    
    assert chat_id == '507f1f77bcf86cd799439013'
    mock_mongo.chat_interactions.insert_one.assert_called_once()
    
    # Verify chat data structure
    call_args = mock_mongo.chat_interactions.insert_one.call_args[0][0]
    assert call_args['user_id'] == 'test_user'
    assert call_args['subject'] == 'mathematics'
    assert call_args['topic'] == 'geometry'
    assert 'sentiment' in call_args

def test_end_session(mock_mongo):
    """Test ending a learning session"""
    tracker = StudentTracker()
    
    # Mock active session
    mock_session = {
        '_id': '507f1f77bcf86cd799439011',
        'user_id': 'test_user',
        'start_time': datetime.now(),
        'activities_count': 5,
        'chat_messages_count': 3,
        'learning_objectives_met': ['spatial_reasoning'],
        'status': 'active'
    }
    mock_mongo.student_sessions.find_one.return_value = mock_session
    
    summary = tracker.end_session('test_user')
    
    assert summary is not None
    assert summary['activities_count'] == 5
    assert summary['chat_messages_count'] == 3
    assert 'duration_minutes' in summary
    
    # Verify session was updated
    mock_mongo.student_sessions.update_one.assert_called_once()

def test_end_session_no_active_session(mock_mongo):
    """Test ending session when no active session exists"""
    tracker = StudentTracker()
    
    mock_mongo.student_sessions.find_one.return_value = None
    
    summary = tracker.end_session('test_user')
    
    assert summary is None

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    with patch('app.main.MongoClient'):
        tracker = StudentTracker()
        
        # Test positive sentiment
        positive_sentiment = tracker._analyze_sentiment("This is great and fun!")
        assert positive_sentiment == 'positive'
        
        # Test negative sentiment  
        negative_sentiment = tracker._analyze_sentiment("This is bad and difficult")
        assert negative_sentiment == 'negative'
        
        # Test neutral sentiment
        neutral_sentiment = tracker._analyze_sentiment("This is a statement")
        assert neutral_sentiment == 'neutral'

def test_get_student_analytics(mock_mongo):
    """Test getting student analytics"""
    tracker = StudentTracker()
    
    # Mock data
    mock_sessions = [{
        'user_id': 'test_user',
        'duration_minutes': 30,
        'learning_objectives_met': ['spatial_reasoning']
    }]
    mock_activities = [{
        'user_id': 'test_user',
        'activity_type': 'building',
        'timestamp': datetime.now()
    }]
    mock_chats = [{
        'user_id': 'test_user', 
        'subject': 'mathematics',
        'sentiment': 'positive'
    }]
    
    mock_mongo.student_sessions.find.return_value = mock_sessions
    mock_mongo.student_activities.find.return_value = mock_activities
    mock_mongo.chat_interactions.find.return_value = mock_chats
    mock_mongo.student_sessions.distinct.return_value = ['spatial_reasoning']
    
    analytics = tracker.get_student_analytics('test_user', 7)
    
    assert analytics['user_id'] == 'test_user'
    assert analytics['period_days'] == 7
    assert 'summary' in analytics
    assert 'engagement' in analytics
    assert 'progress' in analytics
    
    # Check summary data
    assert analytics['summary']['total_sessions'] == 1
    assert analytics['summary']['total_time_minutes'] == 30
    assert analytics['summary']['total_activities'] == 1
    assert analytics['summary']['total_chat_messages'] == 1