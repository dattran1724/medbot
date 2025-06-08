"""
Tools for interacting with the MedicalCare API.
"""

import requests
from langchain_core.tools import tool
from config import API_BASE_URL
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='medical_care.log'
)
logger = logging.getLogger('api_tools')

@tool
def get_available_departments():
    """
    Retrieve a list of all available medical departments/specialties.
    
    Returns:
        A list of departments with their details
    """
    try:
        response = requests.get(f"{API_BASE_URL}?action=get_chuyenkhoa")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return f"Available departments: {data.get('data')}"
            else:
                logger.warning(f"Failed to get departments: {data.get('message', 'No error message')}")
                return "Could not retrieve departments at the moment."
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in get_available_departments")
        return f"An error occurred: {str(e)}"

@tool
def get_department_details(department_id: str):
    """
    Get detailed information about a specific medical department and its doctors.
    
    Args:
        department_id: The ID of the department to get information about
        
    Returns:
        Detailed information about the department and its doctors
    """
    try:
        response = requests.get(f"{API_BASE_URL}?action=get_chitiet_chuyenkhoa&id={department_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return f"Department details: {data.get('data')}"
            else:
                logger.warning(f"Department not found: {department_id}")
                return "Department not found or information unavailable."
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in get_department_details")
        return f"An error occurred: {str(e)}"

@tool
def find_doctors(specialty: str = None):
    """
    Find doctors based on specialty.
    
    Args:
        specialty: The specialty or department to search for doctors
        
    Returns:
        A list of doctors matching the criteria
    """
    try:
        response = requests.get(f"{API_BASE_URL}?action=get_chuyengia")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                doctors = data.get('data')
                # Filter by specialty if provided
                if specialty and specialty.strip():
                    filtered_doctors = [doc for doc in doctors if specialty.lower() in doc.get('specialty', '').lower()]
                    return f"Found doctors with specialty '{specialty}': {filtered_doctors}"
                return f"Found doctors: {doctors}"
            else:
                logger.warning("No doctors found")
                return "No doctors found matching your criteria."
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in find_doctors")
        return f"An error occurred: {str(e)}"

@tool
def get_doctor_details(doctor_id: str):
    """
    Get detailed information about a specific doctor.
    
    Args:
        doctor_id: The ID of the doctor to get details about
        
    Returns:
        Detailed information about the doctor
    """
    try:
        response = requests.get(f"{API_BASE_URL}?action=get_chitiet_chuyengia&id={doctor_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return f"Doctor details: {data.get('data')}"
            else:
                logger.warning(f"Doctor not found: {doctor_id}")
                return "Doctor not found or information unavailable."
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in get_doctor_details")
        return f"An error occurred: {str(e)}"

@tool
def get_doctor_available_times(doctor_id: str, date: str):
    """
    Get available time slots for a doctor.
    
    Args:
        doctor_id: The ID of the doctor to check availability
        date: The date to check availability for (format: YYYY-MM-DD)
        
    Returns:
        A list of available time slots for the specified doctor and date
    """
    try:
        response = requests.get(f"{API_BASE_URL}?action=get_all_schedule_by_doctor&id_bs={doctor_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                # Filter by date if available
                all_times = data.get('data', [])
                if date and date.strip():
                    filtered_times = [slot for slot in all_times if date in slot.get('date', '')]
                    return f"Available times on {date}: {filtered_times}"
                return f"Available times: {all_times}"
            else:
                logger.warning(f"No available time slots for doctor: {doctor_id}")
                return "No available time slots found for this doctor on the specified date."
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in get_doctor_available_times")
        return f"An error occurred: {str(e)}"

@tool
def book_appointment(patient_id: str, doctor_id: str, schedule_id: str, reason: str):
    """
    Book an appointment with a doctor.
    
    Args:
        patient_id: The ID of the patient booking the appointment
        doctor_id: The ID of the doctor
        schedule_id: The ID of the selected time slot
        reason: The reason for the appointment
        
    Returns:
        Confirmation of the appointment booking
    """
    try:
        payload = {
            "id_bn": patient_id,
            "id_bs": doctor_id,
            "id_kh": schedule_id,
            "lydo": reason
        }
        response = requests.post(f"{API_BASE_URL}?action=add_lichhen", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                logger.info(f"Appointment booked successfully for patient {patient_id} with doctor {doctor_id}")
                return f"Appointment booked successfully: {data.get('message')}"
            else:
                logger.warning(f"Failed to book appointment: {data.get('message')}")
                return f"Failed to book appointment: {data.get('message')}"
        else:
            logger.error(f"API error: {response.status_code}")
            return "Failed to connect to the server. Please try again later."
    except Exception as e:
        logger.exception("Exception in book_appointment")
        return f"An error occurred: {str(e)}"

# List all available tools for easy import
available_tools = [
    get_available_departments,
    get_department_details,
    find_doctors,
    get_doctor_details,
    get_doctor_available_times,
    book_appointment,
]
