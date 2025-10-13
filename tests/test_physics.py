import numpy as np
import pytest
from structural import StructuralTireModel, StructuralParams, TireConstruction, ContactPatchShape
from aerodynamic import AerodynamicModel, AeroParams, AeroMode, CarPosition
from multiphysics import MultiPhysicsCoupling, MultiPhysicsParams, CouplingMode
from compound import AdvancedCompoundModel, CompoundParams, CompoundType, MaterialProperty
from thermal import ThermalModel, ThermalParams
from wear import TireWearModel, WearParams
from weather import WeatherModel, WeatherParams

def test_structural_model_initialization():
    """Test structural tire model initialization."""
    params = StructuralParams()
    model = StructuralTireModel(params, TireConstruction.RADIAL)
    
    assert model.p.tire_width == 245.0
    assert model.p.aspect_ratio == 0.4
    assert model.construction == TireConstruction.RADIAL
    assert model.current_deflection == 0.0
    assert model.contact_patch_area == 0.0

def test_structural_deflection_calculation():
    """Test tire deflection calculation."""
    model = StructuralTireModel()
    
    # Test normal conditions
    deflection = model.calculate_deflection(4000.0, 1.5, 85.0, 0.0)
    assert 0.0 <= deflection <= model.p.max_deflection
    
    # Test high load conditions
    high_deflection = model.calculate_deflection(6000.0, 1.2, 85.0, 0.0)
    assert high_deflection > deflection
    
    # Test high temperature (reduced stiffness)
    hot_deflection = model.calculate_deflection(4000.0, 1.5, 120.0, 0.0)
    assert hot_deflection > deflection

def test_structural_contact_patch():
    """Test contact patch calculation."""
    model = StructuralTireModel()
    
    deflection = 0.1
    vertical_load = 4000.0
    slip_angle = 0.05
    
    contact_patch = model.calculate_contact_patch(deflection, vertical_load, slip_angle)
    
    assert 'length' in contact_patch
    assert 'width' in contact_patch
    assert 'area' in contact_patch
    assert 'average_pressure' in contact_patch
    assert 'shape' in contact_patch
    
    assert contact_patch['area'] > 0
    assert contact_patch['average_pressure'] > 0

def test_structural_pressure_distribution():
    """Test pressure distribution calculation."""
    model = StructuralTireModel()
    
    contact_patch = {
        'length': 0.2,
        'width': 0.15,
        'area': 0.03,
        'average_pressure': 1.5
    }
    
    pressure_dist = model.calculate_pressure_distribution(contact_patch, 0.05, 0.02)
    
    assert pressure_dist.shape == (10, 10)
    assert np.all(pressure_dist >= 0)
    assert np.sum(pressure_dist) > 0

def test_structural_forces():
    """Test structural forces calculation."""
    model = StructuralTireModel()
    
    deflection = 0.1
    deflection_velocity = 0.01
    vertical_load = 4000.0
    
    forces = model.calculate_structural_forces(deflection, deflection_velocity, vertical_load)
    
    assert 'spring_force' in forces
    assert 'damping_force' in forces
    assert 'total_structural_force' in forces
    assert 'frequency_response' in forces
    
    assert forces['spring_force'] > 0
    assert forces['total_structural_force'] > 0

def test_structural_state_update():
    """Test structural state update."""
    model = StructuralTireModel()
    
    model.update_structural_state(4000.0, 1.5, 85.0, 0.1, 0.05, 0.02, 0.1)
    
    assert model.current_deflection > 0
    assert model.contact_patch_area > 0
    assert len(model.deflection_history) > 0
    assert len(model.pressure_history) > 0

def test_aerodynamic_model_initialization():
    """Test aerodynamic model initialization."""
    params = AeroParams()
    model = AerodynamicModel(params)
    
    assert model.p.base_downforce == 1.0
    assert model.p.base_cooling == 1.0
    assert model.current_aero_mode == AeroMode.CLEAN_AIR
    assert model.car_position == CarPosition.ISOLATED

def test_aerodynamic_state_update():
    """Test aerodynamic state update."""
    model = AerodynamicModel()
    
    model.update_aerodynamic_state(50.0, 20.0, 0.0, False, 10.0, 45.0)
    
    assert model.car_speed == 50.0
    assert model.following_distance == 20.0
    assert model.wind_speed == 10.0
    assert model.wind_direction == 45.0

def test_aerodynamic_mode_determination():
    """Test aerodynamic mode determination."""
    model = AerodynamicModel()
    
    # Test wake mode
    model.update_aerodynamic_state(50.0, 15.0, 0.0, False, 0.0, 0.0)
    assert model.current_aero_mode == AeroMode.WAKE
    
    # Test DRS mode
    model.update_aerodynamic_state(50.0, 0.0, 0.0, True, 0.0, 0.0)
    assert model.current_aero_mode == AeroMode.DRS_ACTIVE
    
    # Test slipstream mode
    model.update_aerodynamic_state(50.0, 30.0, 0.0, False, 0.0, 0.0)
    assert model.current_aero_mode == AeroMode.SLIPSTREAM

def test_aerodynamic_multipliers():
    """Test aerodynamic multiplier calculation."""
    model = AerodynamicModel()
    
    model.update_aerodynamic_state(50.0, 20.0, 0.0, False, 0.0, 0.0)
    
    effects = model.get_aerodynamic_effects()
    
    assert 'downforce_multiplier' in effects
    assert 'cooling_multiplier' in effects
    assert 'drag_multiplier' in effects
    assert 'aero_mode' in effects
    
    assert effects['downforce_multiplier'] > 0
    assert effects['cooling_multiplier'] > 0
    assert effects['drag_multiplier'] > 0

def test_aerodynamic_tire_cooling():
    """Test tire-specific cooling calculation."""
    model = AerodynamicModel()
    
    model.update_aerodynamic_state(50.0, 20.0, 0.0, False, 0.0, 0.0)
    
    base_cooling = 1.0
    cooling_effect = model.calculate_tire_cooling_effect(base_cooling, "FL")
    
    assert cooling_effect > 0
    assert isinstance(cooling_effect, float)

def test_multiphysics_coupling_initialization():
    """Test multi-physics coupling initialization."""
    thermal_model = ThermalModel(ThermalParams())
    structural_model = StructuralTireModel()
    aerodynamic_model = AerodynamicModel()
    
    coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model)
    
    assert coupling.coupling_mode == CouplingMode.FULL_COUPLING
    assert coupling.coupling_active == True
    assert coupling.p.thermal_structural_coupling == 0.3

def test_multiphysics_coupling_update():
    """Test multi-physics coupling update."""
    thermal_model = ThermalModel(ThermalParams())
    structural_model = StructuralTireModel()
    aerodynamic_model = AerodynamicModel()
    
    coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model)
    
    thermal_state = np.array([100.0, 95.0, 90.0])
    structural_state = {'deflection_ratio': 0.1}
    aerodynamic_state = {'cooling_multiplier': 1.2}
    
    coupling.update_coupling(thermal_state, structural_state, aerodynamic_state, 0.1, 4000.0, 1.5, 0.1)
    
    assert len(coupling.coupling_history) > 0

def test_multiphysics_coupling_modes():
    """Test different coupling modes."""
    thermal_model = ThermalModel(ThermalParams())
    structural_model = StructuralTireModel()
    aerodynamic_model = AerodynamicModel()
    
    coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model)
    
    # Test thermal-structural coupling
    coupling.set_coupling_mode(CouplingMode.THERMAL_STRUCTURAL)
    assert coupling.p.thermal_aero_coupling == 0.0
    assert coupling.p.structural_aero_coupling == 0.0
    
    # Test full coupling
    coupling.set_coupling_mode(CouplingMode.FULL_COUPLING)
    assert coupling.p.thermal_structural_coupling == 0.3
    assert coupling.p.thermal_aero_coupling == 0.1

def test_compound_model_initialization():
    """Test advanced compound model initialization."""
    params = CompoundParams(compound_type=CompoundType.SOFT)
    model = AdvancedCompoundModel(params)
    
    assert model.p.compound_type == CompoundType.SOFT
    assert model.current_temperature == 25.0
    assert model.current_pressure == 1.5
    assert model.current_wear == 0.0

def test_compound_material_properties():
    """Test compound material properties."""
    model = AdvancedCompoundModel()
    
    # Test thermal properties
    thermal_props = model.get_thermal_properties()
    assert 'thermal_conductivity' in thermal_props
    assert 'thermal_capacity' in thermal_props
    assert 'optimal_temp_range' in thermal_props
    
    # Test mechanical properties
    mechanical_props = model.get_mechanical_properties()
    assert 'elastic_modulus' in mechanical_props
    assert 'static_friction' in mechanical_props
    assert 'dynamic_friction' in mechanical_props

def test_compound_state_update():
    """Test compound state update."""
    model = AdvancedCompoundModel()
    
    model.update_compound_state(90.0, 1.6, 0.2, 10.0, 0.1)
    
    assert model.current_temperature == 90.0
    assert model.current_pressure == 1.6
    assert model.current_wear == 0.2
    assert model.ageing_level > 0

def test_compound_temperature_effects():
    """Test temperature effects on compound properties."""
    model = AdvancedCompoundModel()
    
    # Test optimal temperature
    model.update_compound_state(90.0, 1.5, 0.0, 0.0, 0.1)
    optimal_props = model.material_properties.copy()
    
    # Test high temperature
    model.update_compound_state(120.0, 1.5, 0.0, 0.0, 0.1)
    hot_props = model.material_properties.copy()
    
    # Properties should change with temperature
    assert hot_props['thermal_conductivity'] != optimal_props['thermal_conductivity']

def test_compound_pressure_effects():
    """Test pressure effects on compound properties."""
    model = AdvancedCompoundModel()
    
    # Test optimal pressure
    model.update_compound_state(90.0, 1.5, 0.0, 0.0, 0.1)
    optimal_props = model.material_properties.copy()
    
    # Test high pressure
    model.update_compound_state(90.0, 2.0, 0.0, 0.0, 0.1)
    high_pressure_props = model.material_properties.copy()
    
    # Properties should change with pressure
    assert high_pressure_props['elastic_modulus'] != optimal_props['elastic_modulus']

def test_compound_wear_effects():
    """Test wear effects on compound properties."""
    model = AdvancedCompoundModel()
    
    # Test new tire
    model.update_compound_state(90.0, 1.5, 0.0, 0.0, 0.1)
    new_props = model.material_properties.copy()
    
    # Test worn tire
    model.update_compound_state(90.0, 1.5, 0.8, 0.0, 0.1)
    worn_props = model.material_properties.copy()
    
    # Worn tire should have reduced properties
    assert worn_props['wear_resistance'] < new_props['wear_resistance']

def test_compound_recommendations():
    """Test compound recommendations."""
    model = AdvancedCompoundModel()
    
    thermal_state = np.array([110.0, 105.0, 100.0])
    structural_state = {'deflection_ratio': 0.12}
    
    recommendations = model.get_compound_recommendations(thermal_state, structural_state)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_compound_type_characteristics():
    """Test different compound type characteristics."""
    # Test soft compound
    soft_model = AdvancedCompoundModel(CompoundParams(compound_type=CompoundType.SOFT))
    soft_char = soft_model.compound_characteristics
    
    # Test hard compound
    hard_model = AdvancedCompoundModel(CompoundParams(compound_type=CompoundType.HARD))
    hard_char = hard_model.compound_characteristics
    
    # Soft should have higher friction, lower wear resistance
    assert soft_char['static_friction'] > hard_char['static_friction']
    assert soft_char['wear_resistance'] < hard_char['wear_resistance']

def test_compound_summary():
    """Test compound summary generation."""
    model = AdvancedCompoundModel()
    model.update_compound_state(90.0, 1.5, 0.2, 5.0, 0.1)
    
    summary = model.get_compound_summary()
    
    assert 'compound_name' in summary
    assert 'compound_type' in summary
    assert 'current_temperature' in summary
    assert 'thermal_properties' in summary
    assert 'mechanical_properties' in summary
    assert 'wear_properties' in summary

def test_structural_recommendations():
    """Test structural recommendations."""
    model = StructuralTireModel()
    model.update_structural_state(4000.0, 1.5, 110.0, 0.6, 0.05, 0.02, 0.1)
    
    thermal_state = np.array([110.0, 105.0, 100.0])
    wear_level = 0.6
    
    recommendations = model.get_structural_recommendations(thermal_state, wear_level)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_aerodynamic_recommendations():
    """Test aerodynamic recommendations."""
    model = AerodynamicModel()
    model.update_aerodynamic_state(50.0, 15.0, 0.0, True, 20.0, 45.0)
    
    thermal_state = np.array([105.0, 100.0, 95.0])
    
    recommendations = model.get_aerodynamic_recommendations(thermal_state)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_multiphysics_recommendations():
    """Test multi-physics coupling recommendations."""
    thermal_model = ThermalModel(ThermalParams())
    structural_model = StructuralTireModel()
    aerodynamic_model = AerodynamicModel()
    
    coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model)
    
    thermal_state = np.array([110.0, 105.0, 100.0])
    structural_state = {'deflection_ratio': 0.12}
    aerodynamic_state = {'cooling_multiplier': 0.8}
    
    recommendations = coupling.get_coupling_recommendations(thermal_state, structural_state, aerodynamic_state)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_contact_patch_shapes():
    """Test contact patch shape determination."""
    model = StructuralTireModel()
    
    # Test elliptical shape
    contact_patch = model.calculate_contact_patch(0.05, 4000.0, 0.01)
    assert contact_patch['shape'] == ContactPatchShape.ELLIPTICAL.value
    
    # Test rectangular shape
    contact_patch = model.calculate_contact_patch(0.12, 4000.0, 0.01)
    assert contact_patch['shape'] == ContactPatchShape.RECTANGULAR.value
    
    # Test irregular shape
    contact_patch = model.calculate_contact_patch(0.05, 4000.0, 0.15)
    assert contact_patch['shape'] == ContactPatchShape.IRREGULAR.value

def test_aerodynamic_wind_effects():
    """Test wind effects on aerodynamic properties."""
    model = AerodynamicModel()
    
    # Test headwind
    model.update_aerodynamic_state(50.0, 0.0, 0.0, False, 20.0, 0.0)
    headwind_effects = model.get_aerodynamic_effects()
    
    # Test tailwind
    model.update_aerodynamic_state(50.0, 0.0, 0.0, False, 20.0, 180.0)
    tailwind_effects = model.get_aerodynamic_effects()
    
    # Test crosswind
    model.update_aerodynamic_state(50.0, 0.0, 0.0, False, 20.0, 90.0)
    crosswind_effects = model.get_aerodynamic_effects()
    
    # Effects should be different for different wind directions
    assert headwind_effects['wind_direction'] != tailwind_effects['wind_direction']
    assert crosswind_effects['wind_direction'] != headwind_effects['wind_direction']

def test_compound_ageing():
    """Test compound ageing effects."""
    model = AdvancedCompoundModel()
    
    # Test new compound
    model.update_compound_state(90.0, 1.5, 0.0, 0.0, 0.1)
    new_props = model.material_properties.copy()
    
    # Test aged compound
    model.update_compound_state(90.0, 1.5, 0.0, 100.0, 0.1)
    aged_props = model.material_properties.copy()
    
    # Aged compound should have reduced properties
    assert aged_props['static_friction'] < new_props['static_friction']

def test_structural_construction_effects():
    """Test tire construction effects on structural properties."""
    radial_model = StructuralTireModel(StructuralParams(), TireConstruction.RADIAL)
    bias_ply_model = StructuralTireModel(StructuralParams(), TireConstruction.BIAS_PLY)
    
    # Test deflection calculation
    radial_deflection = radial_model.calculate_deflection(4000.0, 1.5, 85.0, 0.0)
    bias_deflection = bias_ply_model.calculate_deflection(4000.0, 1.5, 85.0, 0.0)
    
    # Bias-ply should be stiffer (less deflection)
    assert bias_deflection < radial_deflection

def test_aerodynamic_distance_effects():
    """Test distance effects on aerodynamic properties."""
    model = AerodynamicModel()
    
    # Test close following
    model.update_aerodynamic_state(50.0, 10.0, 0.0, False, 0.0, 0.0)
    close_effects = model.get_aerodynamic_effects()
    
    # Test far following
    model.update_aerodynamic_state(50.0, 40.0, 0.0, False, 0.0, 0.0)
    far_effects = model.get_aerodynamic_effects()
    
    # Effects should be different for different distances
    assert close_effects['following_distance'] != far_effects['following_distance']

def test_multiphysics_coupling_strength():
    """Test coupling strength effects."""
    thermal_model = ThermalModel(ThermalParams())
    structural_model = StructuralTireModel()
    aerodynamic_model = AerodynamicModel()
    
    # Test weak coupling
    weak_params = MultiPhysicsParams(thermal_structural_coupling=0.1)
    weak_coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model, weak_params)
    
    # Test strong coupling
    strong_params = MultiPhysicsParams(thermal_structural_coupling=0.5)
    strong_coupling = MultiPhysicsCoupling(thermal_model, structural_model, aerodynamic_model, strong_params)
    
    assert weak_coupling.p.thermal_structural_coupling < strong_coupling.p.thermal_structural_coupling

def test_compound_reset():
    """Test compound state reset."""
    model = AdvancedCompoundModel()
    model.update_compound_state(90.0, 1.5, 0.2, 10.0, 0.1)
    
    assert model.current_temperature == 90.0
    assert model.current_wear == 0.2
    assert model.ageing_level > 0
    
    model.reset_compound_state()
    
    assert model.current_temperature == 25.0
    assert model.current_wear == 0.0
    assert model.ageing_level == 0.0

def test_structural_reset():
    """Test structural state reset."""
    model = StructuralTireModel()
    model.update_structural_state(4000.0, 1.5, 85.0, 0.1, 0.05, 0.02, 0.1)
    
    assert model.current_deflection > 0
    assert len(model.deflection_history) > 0
    
    model.reset_structural_state()
    
    assert model.current_deflection == 0.0
    assert len(model.deflection_history) == 0

def test_aerodynamic_reset():
    """Test aerodynamic state reset."""
    model = AerodynamicModel()
    model.update_aerodynamic_state(50.0, 20.0, 0.0, True, 10.0, 45.0)
    
    assert model.car_speed == 50.0
    assert model.drs_active == True
    assert len(model.aero_history) > 0
    
    model.reset_aerodynamic_state()
    
    assert model.car_speed == 0.0
    assert model.drs_active == False
    assert len(model.aero_history) == 0
