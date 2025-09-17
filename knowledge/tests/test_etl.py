import pytest
from unittest.mock import Mock, patch
from app.etl import get_educational_content, run_etl

def test_get_educational_content():
    """Test that educational content is properly structured"""
    content = get_educational_content()
    
    assert len(content) > 0
    
    # Check first item structure
    item = content[0]
    required_fields = ['id', 'subject', 'topic', 'title', 'content', 'age_group', 'difficulty']
    
    for field in required_fields:
        assert field in item, f"Missing required field: {field}"
    
    # Check content quality
    assert len(item['content']) > 50, "Content should be substantial"
    assert item['age_group'] in ['10-15', '11-15', '12-15'], "Invalid age group"
    assert item['difficulty'] in ['beginner', 'intermediate', 'advanced'], "Invalid difficulty"

def test_subjects_coverage():
    """Test that we have content for core subjects"""
    content = get_educational_content()
    subjects = [item['subject'] for item in content]
    
    assert 'mathematics' in subjects, "Missing mathematics content"
    assert 'physics' in subjects, "Missing physics content"

@patch('app.etl.Elasticsearch')
def test_run_etl_success(mock_es_class):
    """Test successful ETL run"""
    # Mock Elasticsearch instance
    mock_es = Mock()
    mock_es.ping.return_value = True
    mock_es.indices.exists.return_value = False
    mock_es.count.return_value = {'count': 3}
    mock_es_class.return_value = mock_es
    
    result = run_etl()
    
    assert result == "success"
    mock_es.ping.assert_called_once()
    mock_es.indices.create.assert_called_once()

@patch('app.etl.Elasticsearch')
def test_run_etl_connection_failure(mock_es_class):
    """Test ETL with Elasticsearch connection failure"""
    # Mock failed connection
    mock_es = Mock()
    mock_es.ping.return_value = False
    mock_es_class.return_value = mock_es
    
    result = run_etl()
    
    assert result == "error"