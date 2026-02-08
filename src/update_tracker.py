from datetime import datetime, timedelta
from .models.similarity_model import SimilarityModel

class UpdateTracker:
    def __init__(self):
        self.similarity_model = SimilarityModel()
        self.update_history = []
        
    def track_policy_update(self, old_policy, new_policy, policy_id=None, version=None):
        changes = self.similarity_model.detect_changes(old_policy, new_policy)
        similarity_score = self.similarity_model.calculate_overall_similarity(old_policy, new_policy)
        
        update_record = {
            'policy_id': policy_id,
            'old_version': version - 1 if version else None,
            'new_version': version,
            'update_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'similarity_score': float(similarity_score),
            'total_changes': len(changes),
            'significant_changes': len([c for c in changes if c['similarity'] < 0.7]),
            'changes_detail': changes
        }
        
        self.update_history.append(update_record)
        
        notifications = self._generate_notifications(update_record)
        
        return {
            'update_summary': update_record,
            'notifications': notifications
        }
    
    def _generate_notifications(self, update_record):
        notifications = []
        
        if update_record['similarity_score'] < 0.5:
            notifications.append({
                'type': 'major_change',
                'message': 'Major policy update detected',
                'priority': 'high',
                'recipients': ['compliance_officer', 'legal_team', 'management']
            })
        
        for change in update_record['changes_detail']:
            if change['change_type'] == 'removed' and change['section'] in ['user rights', 'security', 'data protection']:
                notifications.append({
                    'type': 'critical_removal',
                    'message': f"Critical section '{change['section']}' was removed",
                    'priority': 'critical',
                    'recipients': ['compliance_officer', 'legal_team']
                })
            
            if change['change_type'] == 'modified' and change['similarity'] < 0.3:
                notifications.append({
                    'type': 'significant_modification',
                    'message': f"Section '{change['section']}' was significantly modified",
                    'priority': 'medium',
                    'recipients': ['compliance_officer']
                })
        
        if update_record['total_changes'] > 10:
            notifications.append({
                'type': 'multiple_changes',
                'message': f"Policy has {update_record['total_changes']} changes",
                'priority': 'medium',
                'recipients': ['compliance_officer']
            })
        
        return notifications
    
    def get_update_history(self, policy_id=None, days_back=30):
        filtered_history = self.update_history
        
        if policy_id:
            filtered_history = [h for h in filtered_history if h['policy_id'] == policy_id]
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_history = []
        for record in filtered_history:
            record_date = datetime.strptime(record['update_timestamp'], '%Y-%m-%d %H:%M:%S')
            if record_date >= cutoff_date:
                recent_history.append(record)
        
        return sorted(recent_history, key=lambda x: x['update_timestamp'], reverse=True)
    
    def find_similar_updates(self, current_update, similarity_threshold=0.8):
        similar_updates = []
        
        for historical_update in self.update_history:
            if historical_update['policy_id'] == current_update['policy_id']:
                continue
            
            sim_score = self.similarity_model.calculate_overall_similarity(
                str(historical_update['changes_detail']),
                str(current_update['changes_detail'])
            )
            
            if sim_score >= similarity_threshold:
                similar_updates.append({
                    'historical_update': historical_update,
                    'similarity_score': float(sim_score)
                })
        
        return sorted(similar_updates, key=lambda x: x['similarity_score'], reverse=True)