<?php
$act = isset($_GET['act']) ? $_GET['act'] : 'home';
$process = isset($_GET['process']) ? $_GET['process'] : '';
switch ($act) {
  case 'home':
    require_once('home.php');
    break;
  case 'account':
    switch ($process) {
      case 'edit':
        require_once('account/account_edit.php');
        break;
      case 'update':
      default:
        require_once('account/account.php');
        break;
    }
    break;
  case 'order':
    switch ($process) {
      case 'detail':
        require_once('order/order-detail.php');
        break;
      case 'accept':
      default:
        require_once('order/order.php');
        break;
    }
    break;
  case 'news':
    switch ($process) {
      case 'add':
        require_once('news/news_add.php');
        break;
      case 'edit':
        require_once('news/news_edit.php');
        break;
      case 'store':
      case 'update':
      case 'delete':
      default:
        require_once('news/news.php');
        break;
    }
    break;
  case 'drug':
    switch ($process) {
      case 'add':
        require_once('drug/drug_add.php');
        break;
      case 'edit':
        require_once('drug/drug_edit.php');
        break;
      case 'store':
      case 'update':
      case 'delete':
      default:
        require_once('drug/drug.php');
        break;
    }
    break;
  case 'category':
    switch ($process) {
      case 'add':
        require_once('category/category_add.php');
        break;
      case 'edit':
        require_once('category/category_edit.php');
        break;
      case 'store':
      case 'update':
      case 'delete':
      default:
        require_once('category/category.php');
        break;
    }
    break;
  case 'image_control':
    require_once('image_control/control.php');
    break;
  case 'doctor':
    switch ($process) {
      case 'list':
        require_once('doctor/doctor.php');
        break;
      case 'add':
        require_once('doctor/add.php');
        break;
      case 'edit':
        require_once('doctor/edit.php');
        break;
      default:
        require_once('doctor/doctor.php');
        break;
    }
    break;
  case 'appointment':
    switch ($process) {
      case 'list':
        require_once('appointment/appointment.php');
        break;
      case 'detail':
        require_once('appointment/detail.php');
        break;
      default:
        require_once('appointment/appointment.php');
        break;
    }
    break;
  case 'specialist':
    switch ($process) {
      case 'list':
        require_once('specialist/specialist.php');
        break;
      case 'add':
        require_once('specialist/add.php');
        break;
      case 'edit':
        require_once('specialist/edit.php');
        break;
      default:
        require_once('specialist/specialist.php');
        break;
    }
    break;
  case 'status':
    switch ($process) {
      case 'list':
        require_once('status/status.php');
        break;
      case 'edit':
        require_once('status/edit.php');
        break;
      default:
        require_once('status/status.php');
        break;
    }
    break;
  case 'review':
    require_once('review/review.php');
    break;
  case 'plan':
    switch ($process) {
      case 'list':
        require_once('plan/plan.php');
        break;
      case 'edit':
        require_once('plan/edit.php');
        break;
      default:
        require_once('plan/plan.php');
        break;
    }
    break;
    case 'patient':
      switch ($process) {
        case 'list':
          require_once('patient/patient.php');
          break;
        case 'edit':
          require_once('patient/edit.php');
          break;
        default:
          require_once('patient/patient.php');
          break;
      }
      break;
  case 'error':
  default:
    require_once("error.php");
    break;
}
