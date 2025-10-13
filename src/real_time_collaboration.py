from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
from collections import defaultdict
import statistics

class CollaborationRole(Enum):
    """Roles in the collaboration system."""
    RACE_ENGINEER = "race_engineer"
    STRATEGY_ENGINEER = "strategy_engineer"
    TIRE_ENGINEER = "tire_engineer"
    DRIVER = "driver"
    TEAM_PRINCIPAL = "team_principal"
    DATA_ANALYST = "data_analyst"
    WEATHER_ANALYST = "weather_analyst"
    OBSERVER = "observer"

class DecisionType(Enum):
    """Types of decisions in the collaboration system."""
    PIT_STOP = "pit_stop"
    COMPOUND_CHANGE = "compound_change"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    DRIVING_STYLE = "driving_style"
    TIRE_PRESSURE = "tire_pressure"
    FUEL_STRATEGY = "fuel_strategy"
    WEATHER_RESPONSE = "weather_response"
    EMERGENCY_ACTION = "emergency_action"

class DecisionStatus(Enum):
    """Status of decisions."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Priority levels for decisions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class CollaborationParams:
    """Parameters for real-time collaboration."""
    # Session parameters
    session_timeout: int = 3600  # seconds
    max_participants: int = 20
    decision_timeout: int = 300  # seconds
    
    # Decision parameters
    approval_threshold: float = 0.6  # Percentage of participants needed
    critical_approval_threshold: float = 0.8
    auto_approve_timeout: int = 60  # seconds for critical decisions
    
    # Communication parameters
    message_history_size: int = 1000
    notification_enabled: bool = True
    real_time_updates: bool = True
    
    # Security parameters
    authentication_required: bool = True
    role_based_access: bool = True
    audit_logging: bool = True

class TeamMember:
    """Represents a team member in the collaboration system."""
    
    def __init__(self, user_id: str, name: str, role: CollaborationRole, 
                 permissions: List[str] = None):
        self.user_id = user_id
        self.name = name
        self.role = role
        self.permissions = permissions or []
        self.is_online = False
        self.last_activity = datetime.now()
        self.session_id = None
        
        # Decision history
        self.decisions_made = 0
        self.decisions_approved = 0
        self.decisions_rejected = 0
        
        # Performance metrics
        self.response_time_avg = 0.0
        self.accuracy_score = 0.0

class Decision:
    """Represents a decision in the collaboration system."""
    
    def __init__(self, decision_id: str, decision_type: DecisionType, 
                 proposer: TeamMember, description: str, priority: Priority = Priority.MEDIUM):
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.proposer = proposer
        self.description = description
        self.priority = priority
        self.status = DecisionStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Decision details
        self.data = {}
        self.recommendations = []
        self.risks = []
        self.benefits = []
        
        # Voting
        self.votes = {}
        self.approval_count = 0
        self.rejection_count = 0
        self.abstention_count = 0
        
        # Implementation
        self.implemented_by = None
        self.implemented_at = None
        self.implementation_notes = ""
        
        # Analysis
        self.impact_score = 0.0
        self.confidence_score = 0.0
        self.success_probability = 0.0

class CollaborationSession:
    """Represents a collaboration session."""
    
    def __init__(self, session_id: str, session_name: str, 
                 params: CollaborationParams = None):
        self.session_id = session_id
        self.session_name = session_name
        self.p = params or CollaborationParams()
        
        # Session state
        self.created_at = datetime.now()
        self.is_active = True
        self.race_context = {}
        
        # Participants
        self.participants = {}
        self.active_participants = set()
        
        # Decisions
        self.decisions = {}
        self.pending_decisions = []
        self.completed_decisions = []
        
        # Communication
        self.messages = []
        self.notifications = []
        
        # Analytics
        self.session_metrics = {
            'total_decisions': 0,
            'decisions_approved': 0,
            'decisions_rejected': 0,
            'average_response_time': 0.0,
            'collaboration_score': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        self.decision_event = threading.Event()

class RealTimeCollaboration:
    """
    Real-time collaboration system for F1 team decision making.
    
    Features:
    - Multi-user collaboration with role-based access
    - Real-time decision making and voting
    - Priority-based decision handling
    - Automated decision analysis and recommendations
    - Performance tracking and analytics
    - Audit logging and compliance
    - Integration with simulation and optimization engines
    - Mobile and web interface support
    """
    
    def __init__(self, params: CollaborationParams = None):
        self.p = params or CollaborationParams()
        
        # Active sessions
        self.active_sessions = {}
        self.session_lock = threading.Lock()
        
        # User management
        self.registered_users = {}
        self.user_sessions = {}
        
        # Decision engine integration
        self.simulation_engine = None
        self.strategy_optimizer = None
        self.decision_engine = None
        
        # Analytics
        self.collaboration_analytics = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_decisions': 0,
            'average_session_duration': 0.0,
            'user_satisfaction_score': 0.0
        }
        
        # Background tasks
        self.background_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def create_session(self, session_name: str, creator: TeamMember) -> str:
        """
        Create a new collaboration session.
        
        Args:
            session_name: Name of the session
            creator: Team member creating the session
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(session_id, session_name, self.p)
        
        # Add creator as first participant
        session.participants[creator.user_id] = creator
        session.active_participants.add(creator.user_id)
        creator.session_id = session_id
        creator.is_online = True
        
        with self.session_lock:
            self.active_sessions[session_id] = session
        
        # Update analytics
        self.collaboration_analytics['total_sessions'] += 1
        self.collaboration_analytics['active_sessions'] += 1
        
        return session_id
    
    def join_session(self, session_id: str, user: TeamMember) -> bool:
        """
        Join an existing collaboration session.
        
        Args:
            session_id: ID of the session to join
            user: User joining the session
            
        Returns:
            True if successful, False otherwise
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check if session is full
            if len(session.participants) >= self.p.max_participants:
                return False
            
            # Add user to session
            session.participants[user.user_id] = user
            session.active_participants.add(user.user_id)
            user.session_id = session_id
            user.is_online = True
            
            # Send notification to other participants
            self._notify_session_join(session, user)
            
            return True
    
    def leave_session(self, session_id: str, user_id: str):
        """
        Leave a collaboration session.
        
        Args:
            session_id: ID of the session to leave
            user_id: ID of the user leaving
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                return
            
            session = self.active_sessions[session_id]
            
            if user_id in session.participants:
                user = session.participants[user_id]
                user.is_online = False
                user.session_id = None
                
                session.active_participants.discard(user_id)
                
                # Send notification to other participants
                self._notify_session_leave(session, user)
                
                # If no active participants, close session
                if not session.active_participants:
                    self._close_session(session_id)
    
    def propose_decision(self, session_id: str, proposer_id: str, 
                        decision_type: DecisionType, description: str,
                        priority: Priority = Priority.MEDIUM, 
                        data: Dict[str, Any] = None) -> str:
        """
        Propose a new decision.
        
        Args:
            session_id: ID of the session
            proposer_id: ID of the user proposing the decision
            decision_type: Type of decision
            description: Description of the decision
            priority: Priority level
            data: Additional decision data
            
        Returns:
            Decision ID
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
            
            session = self.active_sessions[session_id]
            
            if proposer_id not in session.participants:
                raise ValueError("User not in session")
            
            proposer = session.participants[proposer_id]
            
            # Create decision
            decision_id = str(uuid.uuid4())
            decision = Decision(decision_id, decision_type, proposer, description, priority)
            
            if data:
                decision.data = data
            
            # Analyze decision
            self._analyze_decision(decision, session)
            
            # Add to session
            session.decisions[decision_id] = decision
            session.pending_decisions.append(decision)
            session.session_metrics['total_decisions'] += 1
            
            # Notify participants
            self._notify_decision_proposed(session, decision)
            
            # Start decision timeout
            self._start_decision_timeout(session_id, decision_id)
            
            return decision_id
    
    def vote_on_decision(self, session_id: str, decision_id: str, 
                        voter_id: str, vote: str, comments: str = "") -> bool:
        """
        Vote on a decision.
        
        Args:
            session_id: ID of the session
            decision_id: ID of the decision
            voter_id: ID of the voter
            vote: Vote ('approve', 'reject', 'abstain')
            comments: Optional comments
            
        Returns:
            True if vote was recorded, False otherwise
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            if decision_id not in session.decisions:
                return False
            
            if voter_id not in session.participants:
                return False
            
            decision = session.decisions[decision_id]
            
            # Check if decision is still pending
            if decision.status != DecisionStatus.PENDING:
                return False
            
            # Record vote
            decision.votes[voter_id] = {
                'vote': vote,
                'comments': comments,
                'timestamp': datetime.now()
            }
            
            # Update vote counts
            if vote == 'approve':
                decision.approval_count += 1
            elif vote == 'reject':
                decision.rejection_count += 1
            elif vote == 'abstain':
                decision.abstention_count += 1
            
            # Check if decision can be resolved
            self._check_decision_resolution(session, decision)
            
            # Notify participants
            self._notify_vote_cast(session, decision, voter_id, vote)
            
            return True
    
    def implement_decision(self, session_id: str, decision_id: str, 
                          implementer_id: str, notes: str = "") -> bool:
        """
        Implement a decision.
        
        Args:
            session_id: ID of the session
            decision_id: ID of the decision
            implementer_id: ID of the implementer
            notes: Implementation notes
            
        Returns:
            True if implementation was successful, False otherwise
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            if decision_id not in session.decisions:
                return False
            
            decision = session.decisions[decision_id]
            
            # Check if decision is approved
            if decision.status != DecisionStatus.APPROVED:
                return False
            
            # Implement decision
            decision.status = DecisionStatus.IMPLEMENTED
            decision.implemented_by = implementer_id
            decision.implemented_at = datetime.now()
            decision.implementation_notes = notes
            decision.updated_at = datetime.now()
            
            # Move to completed decisions
            if decision in session.pending_decisions:
                session.pending_decisions.remove(decision)
            session.completed_decisions.append(decision)
            
            # Update metrics
            session.session_metrics['decisions_approved'] += 1
            
            # Execute decision (integrate with simulation/optimization)
            self._execute_decision(decision, session)
            
            # Notify participants
            self._notify_decision_implemented(session, decision)
            
            return True
    
    def _analyze_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze decision and provide recommendations."""
        # Analyze decision based on type
        if decision.decision_type == DecisionType.PIT_STOP:
            self._analyze_pit_stop_decision(decision, session)
        elif decision.decision_type == DecisionType.COMPOUND_CHANGE:
            self._analyze_compound_change_decision(decision, session)
        elif decision.decision_type == DecisionType.STRATEGY_ADJUSTMENT:
            self._analyze_strategy_adjustment_decision(decision, session)
        elif decision.decision_type == DecisionType.DRIVING_STYLE:
            self._analyze_driving_style_decision(decision, session)
        elif decision.decision_type == DecisionType.TIRE_PRESSURE:
            self._analyze_tire_pressure_decision(decision, session)
        elif decision.decision_type == DecisionType.FUEL_STRATEGY:
            self._analyze_fuel_strategy_decision(decision, session)
        elif decision.decision_type == DecisionType.WEATHER_RESPONSE:
            self._analyze_weather_response_decision(decision, session)
        elif decision.decision_type == DecisionType.EMERGENCY_ACTION:
            self._analyze_emergency_action_decision(decision, session)
    
    def _analyze_pit_stop_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze pit stop decision."""
        # Get current race context
        race_context = session.race_context
        
        # Analyze pit stop timing
        current_lap = race_context.get('current_lap', 0)
        total_laps = race_context.get('total_laps', 58)
        
        # Calculate optimal pit window
        optimal_pit_lap = total_laps // 2
        
        # Analyze timing
        if current_lap < optimal_pit_lap - 5:
            decision.risks.append("Early pit stop may require additional stops")
            decision.benefits.append("Fresh tires for longer stint")
        elif current_lap > optimal_pit_lap + 5:
            decision.risks.append("Late pit stop may cause tire degradation")
            decision.benefits.append("Maximize current stint length")
        else:
            decision.benefits.append("Optimal pit stop timing")
        
        # Calculate impact score
        decision.impact_score = 0.7  # High impact for pit stops
        decision.confidence_score = 0.8
        decision.success_probability = 0.75
    
    def _analyze_compound_change_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze compound change decision."""
        # Get weather conditions
        weather = session.race_context.get('weather', {})
        rain_probability = weather.get('rain_probability', 0.0)
        
        # Analyze compound selection
        if rain_probability > 0.5:
            decision.benefits.append("Wet weather compound provides better grip")
            decision.risks.append("Dry weather performance may be compromised")
        else:
            decision.benefits.append("Dry weather compound provides optimal performance")
            decision.risks.append("Wet weather performance may be compromised")
        
        # Calculate impact score
        decision.impact_score = 0.6
        decision.confidence_score = 0.7
        decision.success_probability = 0.8
    
    def _analyze_strategy_adjustment_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze strategy adjustment decision."""
        # Analyze strategy change impact
        decision.benefits.append("Strategy adjustment may improve race position")
        decision.risks.append("Strategy change may backfire")
        
        # Calculate impact score
        decision.impact_score = 0.5
        decision.confidence_score = 0.6
        decision.success_probability = 0.7
    
    def _analyze_driving_style_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze driving style decision."""
        # Analyze driving style change
        decision.benefits.append("Driving style adjustment may improve lap times")
        decision.risks.append("Aggressive driving may increase tire wear")
        
        # Calculate impact score
        decision.impact_score = 0.4
        decision.confidence_score = 0.5
        decision.success_probability = 0.6
    
    def _analyze_tire_pressure_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze tire pressure decision."""
        # Analyze tire pressure change
        decision.benefits.append("Tire pressure adjustment may improve grip")
        decision.risks.append("Incorrect pressure may cause tire failure")
        
        # Calculate impact score
        decision.impact_score = 0.3
        decision.confidence_score = 0.6
        decision.success_probability = 0.7
    
    def _analyze_fuel_strategy_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze fuel strategy decision."""
        # Analyze fuel strategy change
        decision.benefits.append("Fuel strategy adjustment may improve efficiency")
        decision.risks.append("Fuel strategy change may affect race pace")
        
        # Calculate impact score
        decision.impact_score = 0.4
        decision.confidence_score = 0.5
        decision.success_probability = 0.6
    
    def _analyze_weather_response_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze weather response decision."""
        # Analyze weather response
        decision.benefits.append("Weather response may improve safety and performance")
        decision.risks.append("Weather response may be unnecessary")
        
        # Calculate impact score
        decision.impact_score = 0.6
        decision.confidence_score = 0.7
        decision.success_probability = 0.8
    
    def _analyze_emergency_action_decision(self, decision: Decision, session: CollaborationSession):
        """Analyze emergency action decision."""
        # Analyze emergency action
        decision.benefits.append("Emergency action may prevent serious issues")
        decision.risks.append("Emergency action may be unnecessary")
        
        # Calculate impact score
        decision.impact_score = 0.9  # Very high impact for emergency actions
        decision.confidence_score = 0.8
        decision.success_probability = 0.9
    
    def _check_decision_resolution(self, session: CollaborationSession, decision: Decision):
        """Check if decision can be resolved."""
        total_votes = decision.approval_count + decision.rejection_count + decision.abstention_count
        total_participants = len(session.active_participants)
        
        # Check if enough votes have been cast
        if total_votes >= total_participants * 0.5:  # At least 50% participation
            # Check approval threshold
            approval_threshold = self.p.critical_approval_threshold if decision.priority == Priority.CRITICAL else self.p.approval_threshold
            
            if decision.approval_count / total_votes >= approval_threshold:
                decision.status = DecisionStatus.APPROVED
                decision.updated_at = datetime.now()
                
                # Move to completed decisions
                if decision in session.pending_decisions:
                    session.pending_decisions.remove(decision)
                session.completed_decisions.append(decision)
                
                # Notify participants
                self._notify_decision_approved(session, decision)
            
            elif decision.rejection_count / total_votes >= approval_threshold:
                decision.status = DecisionStatus.REJECTED
                decision.updated_at = datetime.now()
                
                # Move to completed decisions
                if decision in session.pending_decisions:
                    session.pending_decisions.remove(decision)
                session.completed_decisions.append(decision)
                
                # Notify participants
                self._notify_decision_rejected(session, decision)
    
    def _start_decision_timeout(self, session_id: str, decision_id: str):
        """Start decision timeout timer."""
        def timeout_handler():
            time.sleep(self.p.decision_timeout)
            
            with self.session_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    if decision_id in session.decisions:
                        decision = session.decisions[decision_id]
                        if decision.status == DecisionStatus.PENDING:
                            # Auto-approve critical decisions
                            if decision.priority == Priority.CRITICAL:
                                decision.status = DecisionStatus.APPROVED
                                decision.updated_at = datetime.now()
                                
                                # Move to completed decisions
                                if decision in session.pending_decisions:
                                    session.pending_decisions.remove(decision)
                                session.completed_decisions.append(decision)
                                
                                # Notify participants
                                self._notify_decision_auto_approved(session, decision)
                            else:
                                decision.status = DecisionStatus.CANCELLED
                                decision.updated_at = datetime.now()
                                
                                # Move to completed decisions
                                if decision in session.pending_decisions:
                                    session.pending_decisions.remove(decision)
                                session.completed_decisions.append(decision)
                                
                                # Notify participants
                                self._notify_decision_cancelled(session, decision)
        
        # Start timeout in background thread
        timeout_thread = threading.Thread(target=timeout_handler)
        timeout_thread.daemon = True
        timeout_thread.start()
    
    def _execute_decision(self, decision: Decision, session: CollaborationSession):
        """Execute decision by integrating with simulation/optimization engines."""
        # This would integrate with the simulation engine and strategy optimizer
        # For now, we'll just log the execution
        
        execution_data = {
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'implemented_by': decision.implemented_by,
            'implemented_at': decision.implemented_at.isoformat(),
            'data': decision.data,
            'notes': decision.implementation_notes
        }
        
        # Log execution
        if self.p.audit_logging:
            self._log_decision_execution(execution_data)
    
    def _log_decision_execution(self, execution_data: Dict[str, Any]):
        """Log decision execution for audit purposes."""
        # This would write to an audit log
        pass
    
    def _notify_session_join(self, session: CollaborationSession, user: TeamMember):
        """Notify participants of session join."""
        notification = {
            'type': 'session_join',
            'user_id': user.user_id,
            'user_name': user.name,
            'role': user.role.value,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_session_leave(self, session: CollaborationSession, user: TeamMember):
        """Notify participants of session leave."""
        notification = {
            'type': 'session_leave',
            'user_id': user.user_id,
            'user_name': user.name,
            'role': user.role.value,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_proposed(self, session: CollaborationSession, decision: Decision):
        """Notify participants of decision proposal."""
        notification = {
            'type': 'decision_proposed',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'proposer': decision.proposer.name,
            'description': decision.description,
            'priority': decision.priority.value,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_vote_cast(self, session: CollaborationSession, decision: Decision, 
                         voter_id: str, vote: str):
        """Notify participants of vote cast."""
        voter = session.participants[voter_id]
        
        notification = {
            'type': 'vote_cast',
            'decision_id': decision.decision_id,
            'voter': voter.name,
            'vote': vote,
            'approval_count': decision.approval_count,
            'rejection_count': decision.rejection_count,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_approved(self, session: CollaborationSession, decision: Decision):
        """Notify participants of decision approval."""
        notification = {
            'type': 'decision_approved',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'description': decision.description,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_rejected(self, session: CollaborationSession, decision: Decision):
        """Notify participants of decision rejection."""
        notification = {
            'type': 'decision_rejected',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'description': decision.description,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_implemented(self, session: CollaborationSession, decision: Decision):
        """Notify participants of decision implementation."""
        notification = {
            'type': 'decision_implemented',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'implemented_by': decision.implemented_by,
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_auto_approved(self, session: CollaborationSession, decision: Decision):
        """Notify participants of auto-approval."""
        notification = {
            'type': 'decision_auto_approved',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'reason': 'timeout_auto_approval',
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _notify_decision_cancelled(self, session: CollaborationSession, decision: Decision):
        """Notify participants of decision cancellation."""
        notification = {
            'type': 'decision_cancelled',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'reason': 'timeout',
            'timestamp': datetime.now().isoformat()
        }
        
        session.notifications.append(notification)
    
    def _close_session(self, session_id: str):
        """Close a collaboration session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.is_active = False
                
                # Update analytics
                self.collaboration_analytics['active_sessions'] -= 1
                
                # Calculate session metrics
                self._calculate_session_metrics(session)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
    
    def _calculate_session_metrics(self, session: CollaborationSession):
        """Calculate session metrics."""
        # Calculate average response time
        response_times = []
        for decision in session.completed_decisions:
            if decision.implemented_at:
                response_time = (decision.implemented_at - decision.created_at).total_seconds()
                response_times.append(response_time)
        
        if response_times:
            session.session_metrics['average_response_time'] = statistics.mean(response_times)
        
        # Calculate collaboration score
        total_decisions = session.session_metrics['total_decisions']
        approved_decisions = session.session_metrics['decisions_approved']
        
        if total_decisions > 0:
            approval_rate = approved_decisions / total_decisions
            session.session_metrics['collaboration_score'] = approval_rate * 100
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        def monitor_sessions():
            while True:
                time.sleep(60)  # Check every minute
                
                with self.session_lock:
                    current_time = datetime.now()
                    
                    # Check for inactive sessions
                    inactive_sessions = []
                    for session_id, session in self.active_sessions.items():
                        if (current_time - session.created_at).total_seconds() > self.p.session_timeout:
                            inactive_sessions.append(session_id)
                    
                    # Close inactive sessions
                    for session_id in inactive_sessions:
                        self._close_session(session_id)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_sessions)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status and metrics."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return {'error': 'Session not found'}
            
            session = self.active_sessions[session_id]
            
            return {
                'session_id': session_id,
                'session_name': session.session_name,
                'is_active': session.is_active,
                'created_at': session.created_at.isoformat(),
                'participants': len(session.participants),
                'active_participants': len(session.active_participants),
                'pending_decisions': len(session.pending_decisions),
                'completed_decisions': len(session.completed_decisions),
                'session_metrics': session.session_metrics,
                'recent_notifications': session.notifications[-10:]  # Last 10 notifications
            }
    
    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration analytics."""
        return {
            'collaboration_analytics': self.collaboration_analytics,
            'active_sessions': len(self.active_sessions),
            'registered_users': len(self.registered_users),
            'total_decisions': sum(session.session_metrics['total_decisions'] for session in self.active_sessions.values()),
            'average_session_duration': self.collaboration_analytics['average_session_duration'],
            'user_satisfaction_score': self.collaboration_analytics['user_satisfaction_score']
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
