<?php
session_start();
if (isset($_SESSION['isLogin'])) {
  if ((isset($_SESSION['isLogin']['admin']) && $_SESSION['isLogin']['admin'] == true) || (isset($_SESSION['isLogin']['doctor']) && $_SESSION['isLogin']['doctor'] == true)) {
    $act = isset($_GET['act']) ? $_GET['act'] : "home";
    $process = isset($_GET['process']) ? $_GET['process'] : 'list';
    switch ($act) {
      case 'home':
        require_once('../admin/controllers/homeController.php');
        $controller_obj = new HomeController();
        $controller_obj->list();
        break;
      case 'account':
        $role = filter_input(INPUT_GET, 'role');
        $process = filter_input(INPUT_GET, 'process');
        require_once('../admin/controllers/accountController.php');
        $controller_obj = new AccountController();
        switch ($role) {
          case 'admin':
            $controller_obj->list(3);
            break;
          case 'customer':
            $controller_obj->list(1);
            break;
          case 'doctor':
            $controller_obj->list(2);
            break;
          default:

            break;
        }
        switch ($process) {
          case 'add':
            $controller_obj->add();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
            case 'delete':
              $controller_obj->delete();
              break;
          default:

            break;
        }
        break;
      case 'logout':
        require_once('../Admin/controllers/accountController.php');
        $controller_obj = new AccountController();
        $controller_obj->logout();
        break;
      case 'order':
        require_once('../Admin/controllers/orderController.php');
        $controller_obj = new OrderController();
        $process = isset($_GET['process']) ? $_GET['process'] : '';
        switch ($process) {
          case 'detail':
            $controller_obj->orderDetail();
            break;
          case 'accept':
            $controller_obj->accept();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
        case 'news':
          require_once('../Admin/controllers/newsController.php');
          $controller_obj = new NewsController();
          $process = isset($_GET['process']) ? $_GET['process'] : '';
          switch ($process) {
            case 'add':
              $controller_obj->add();
              break;
            case 'store':
              $controller_obj->store();
              break;
            case 'edit':
              $controller_obj->edit();
              break;
            case 'update':
              $controller_obj->update();
              break;
            case 'delete':
              $controller_obj->delete();
              break;
            default:
              $controller_obj->list();
              break;
          }
          break;
      case 'drug':
        require_once('../Admin/controllers/drugController.php');
        $controller_obj = new DrugController();
        $process = isset($_GET['process']) ? $_GET['process'] : '';
        switch ($process) {
          case 'add':
            $controller_obj->add();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'category':
        require_once('../Admin/controllers/categoryController.php');
        $controller_obj = new CategoryController();
        $process = isset($_GET['process']) ? $_GET['process'] : '';
        switch ($process) {
          case 'add':
            $controller_obj->add();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'pharmacy':
        require_once('../admin/controllers/accountController.php');
        $controller_obj = new AccountController();
        $controller_obj->goback();
        break;
      case 'image_control';
        require_once('../Admin/views/index.php');
        break;
      case 'doctor':
        require_once('../admin/controllers/doctorController.php');
        $controller_obj = new DoctorController();
        switch ($process) {
          case 'list':
            $controller_obj->list();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'add':
            $controller_obj->add();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'appointment':
        require_once('../admin/controllers/AppointmentController.php');
        $controller_obj = new AppointmentController();
        switch ($process) {
          case 'list':
            $controller_obj->list();
            break;
          case 'detail':
            $controller_obj->detail();
            break;
          case 'accept':
            $controller_obj->accept();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'specialist':
        require_once('../admin/controllers/specialistController.php');
        $controller_obj = new SpecialistController();
        switch ($process) {
          case 'list':
            $controller_obj->list();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'add':
            $controller_obj->add();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'status':
        require_once('../admin/controllers/statusController.php');
        $controller_obj = new StatusController();
        switch ($process) {
          case 'list':
            $controller_obj->list();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
      case 'review':
        require_once('../admin/controllers/reviewController.php');
        $controller_obj = new ReviewController();
        $controller_obj->list();
        break;
      case 'plan':
        require_once('../admin/controllers/planController.php');
        $controller_obj = new PlanController();
        switch ($process) {
          case 'list':
            $controller_obj->list();
            break;
          case 'edit':
            $controller_obj->edit();
            break;
          case 'update':
            $controller_obj->update();
            break;
          case 'store':
            $controller_obj->store();
            break;
          case 'delete':
            $controller_obj->delete();
            break;
          default:
            $controller_obj->list();
            break;
        }
        break;
        case 'patient':
          require_once('../admin/controllers/patientController.php');
          $controller_obj = new PatientController();
          switch ($process) {
            case 'list':
              $controller_obj->list();
              break;
            case 'edit':
              $controller_obj->edit();
              break;
            case 'update':
              $controller_obj->update();
              break;
            case 'delete':
              $controller_obj->delete();
              break;
            default:
              $controller_obj->list();
              break;
          }
          break;
      
      case 'error':
        require_once('./views/index.php');
        break;
      default:
        require_once('../admin/controllers/homeController.php');
        break;
    }
  } else {
    header("Location: views/error.php");
  }
} else {
  header("Location: views/error.php");
}
