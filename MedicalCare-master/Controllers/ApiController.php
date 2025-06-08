<?php
require_once("Models/lichhen.php");
require_once("Models/chuyenkhoa.php");
require_once("Models/chuyengia.php");
require_once("Models/detail.php");

class ApiController
{
    var $lichhen_model;
    var $chuyenkhoa_model;
    var $chuyengia_model;
    var $detail_model;

    public function __construct()
    {
        $this->lichhen_model = new Lichhen();
        $this->chuyenkhoa_model = new Chuyenkhoa();
        $this->chuyengia_model = new Chuyengia();
        $this->detail_model = new Detail();
        header('Content-Type: application/json');
    }

    // Helper function to clean database results (remove numeric indices)
    private function cleanData(array $data): array {
        return array_map(function(array $row): array {
            return array_filter(
                $row,
                fn($key) => !is_numeric($key),
                ARRAY_FILTER_USE_KEY
            );
        }, $data);
    }


    public function add_lichhen_api()
    {
        try {
            // Check if request is POST
            if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            // Get JSON data from request body
            $data = json_decode(file_get_contents('php://input'), true);
            
            if (!isset($data['id_bn']) || !isset($data['id_bs']) || !isset($data['id_kh']) || !isset($data['lydo'])) {
                echo json_encode(['status' => 'error', 'message' => 'Missing required parameters']);
                return;
            }

            $id_bn = $data['id_bn'];
            $id_bs = $data['id_bs'];
            $id_kh = $data['id_kh'];
            $lydo = $data['lydo'];

            $result = $this->lichhen_model->add_lichhen($id_bn, $id_bs, $id_kh, $lydo);

            if ($result) {
                echo json_encode(['status' => 'success', 'message' => 'Đặt lịch thành công']);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'Đặt lịch thất bại']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get departments list
    public function get_chuyenkhoa_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            $data = $this->chuyenkhoa_model->list();

            if ($data) {
                $filtered = array_map(function($item) {
                    return [
                        'name' => $item['name'] ?? null,
                        'hinhanh' => $item['hinhanh'] ?? null,
                    ];
                }, $data);

                echo json_encode(['status' => 'success', 'data' => $filtered]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'No data available']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }
    // API to get department details
    public function get_chitiet_chuyenkhoa_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id'])) {
                echo json_encode(['status' => 'error', 'message' => 'Department ID is required']);
                return;
            }

            $id = $_GET['id'];
            $data = $this->chuyenkhoa_model->chitietchuyenkhoa($id);
            $doctors = $this->chuyenkhoa_model->danhsachbacsi($id);
            
            if ($data) {
                echo json_encode([
                    'status' => 'success', 
                    'data' => [
                        'department' => $this->cleanData($data),
                        'doctors' => $this->cleanData($doctors)
                    ]
                ]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'Department not found']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get doctors/specialists list
    public function get_chuyengia_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            $data = $this->chuyengia_model->list();
            
            if ($data) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($data)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'No data available']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get doctor details
    public function get_chitiet_chuyengia_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id'])) {
                echo json_encode(['status' => 'error', 'message' => 'Doctor ID is required']);
                return;
            }

            $id = $_GET['id'];
            $data = $this->chuyengia_model->chitietbacsi($id);
            
            if ($data) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($data)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'Doctor not found']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get doctor's schedule
    public function get_kehoach_bacsi_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id'])) {
                echo json_encode(['status' => 'error', 'message' => 'Schedule ID is required']);
                return;
            }

            $id = $_GET['id'];
            $data = $this->chuyengia_model->lenlich($id);
            
            if ($data) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($data)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'Schedule not found']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get doctor's schedule by ID
    public function get_all_schedule_by_doctor_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id_bs'])) {
                echo json_encode(['status' => 'error', 'message' => 'Doctor ID is required']);
                return;
            }

            $id_bs = $_GET['id_bs'];
            $data = $this->detail_model->kehoach($id_bs);
            
            if ($data) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($data)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'No schedule found for this doctor']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get available dates for a doctor
    public function get_available_dates_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id_bs'])) {
                echo json_encode(['status' => 'error', 'message' => 'Doctor ID is required']);
                return;
            }

            $id_bs = $_GET['id_bs'];
            $dates = $this->detail_model->ngay($id_bs);
            
            if ($dates) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($dates)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'No available dates found for this doctor']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }

    // API to get available time slots for a specific doctor on a specific date
    public function get_available_times_by_date_api()
    {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
                echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
                return;
            }

            if (!isset($_GET['id_bs']) || !isset($_GET['ngay'])) {
                echo json_encode(['status' => 'error', 'message' => 'Doctor ID and date are required']);
                return;
            }

            $id_bs = $_GET['id_bs'];
            $ngay = $_GET['ngay'];
            $times = $this->detail_model->giotheongay($id_bs, $ngay);
            
            if ($times) {
                echo json_encode(['status' => 'success', 'data' => $this->cleanData($times)]);
            } else {
                echo json_encode(['status' => 'error', 'message' => 'No available times found for this doctor on the specified date']);
            }
        } catch (Exception $e) {
            echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
        }
    }
}
?>    

