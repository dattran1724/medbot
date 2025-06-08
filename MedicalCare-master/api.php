<?php
session_start();

require_once("Controllers/ApiController.php");

$action = isset($_GET['action']) ? $_GET['action'] : 'default';

// Create API controller
$api_controller = new ApiController();

// Handle different API endpoints
switch ($action) {
    case 'add_lichhen':
        $api_controller->add_lichhen_api();
        break;
        
    // Chuyên khoa (Department) endpoints
    case 'get_chuyenkhoa':
        $api_controller->get_chuyenkhoa_api();
        break;
    case 'get_chitiet_chuyenkhoa':
        $api_controller->get_chitiet_chuyenkhoa_api();
        break;
        
    // Chuyên gia/Bác sĩ (Doctor/Specialist) endpoints
    case 'get_chuyengia':
        $api_controller->get_chuyengia_api();
        break;
    case 'get_chitiet_chuyengia':
        $api_controller->get_chitiet_chuyengia_api();
        break;
    case 'get_kehoach_bacsi':
        $api_controller->get_kehoach_bacsi_api();
        break;
        
    // New doctor schedule endpoints
    case 'get_all_schedule_by_doctor':
        $api_controller->get_all_schedule_by_doctor_api();
        break;
    case 'get_available_dates':
        $api_controller->get_available_dates_api();
        break;
    case 'get_available_times_by_date':
        $api_controller->get_available_times_by_date_api();
        break;
        
    default:
        header('Content-Type: application/json');
        echo json_encode(['status' => 'error', 'message' => 'Endpoint not found']);
        break;
}
?>
