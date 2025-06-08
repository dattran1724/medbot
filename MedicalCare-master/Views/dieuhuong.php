<?php
$act = isset($_GET['act']) ? $_GET['act'] : "home";
switch ($act) {
  case "home":
    require_once("home/home.php");
    break;
  case "tintuc":
    require_once("home/home.php");
    break;
  case "chuyengia":
    require_once("chuyengia/chuyengia.php");
    break;
  case "chuyenkhoa":
    require_once("chuyenkhoa/chuyenkhoa.php");
    break;
  case "chitietchuyenkhoa":
    require_once("chuyenkhoa/chitietchuyenkhoa.php");
    break;
  case "detail":
    require_once("detail/detail.php");
    break;
  case "lichhen":
    require_once("lichhen/lichhen.php");
    break;
  case 'xemlichhen':
    require_once('lichhen/dadat.php');
    break;
  case 'pharmacy':
    require_once('pharmacy/pharmacy.php');
    break;
  case 'news':
    require_once('news/news.php');
    break;
  case 'news_detail':
    require_once('news/news_detail.php');
    break;
  case 'product-list':
    require_once('pharmacy/product-list.php');
    break;
  case 'product-detail':
    $act2 = isset($_GET['process']) ? $_GET['process'] : '';
    switch ($act2) {
      case 'list_order':
        require_once('order_detail/order_detail.php');
        break;
      case 'make_order':
        require_once('order_detail/order_notification.php');
        break;
      case 'view_order':
        require_once('order_detail/order_status.php');
        break;
      case 'order_detail':
        require_once('order_detail/order_detail_complete.php');
        break;
      default:
        require_once('pharmacy/product-detail.php');
        break;
    }
    break;
  case 'product-rating':
    require_once('pharmacy/product-detail.php');
    break;
  case 'account':
    $act2 = isset($_GET['process']) ? $_GET['process'] : 'login';
    if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) {
      switch ($act2) {
        case 'account':
          require_once('account/account.php');
          break;
        case 'update':
          require_once('account/account.php');
          break;
        default:
          require_once('account/account.php');
          break;
      }
    } else {
      switch ($act2) {
        case 'login':
          require_once('login/login.php');
          break;
        case 'signup':
          require_once('login/register.php');
          break;
        case 'forgot':
          require_once('login/forgot.php');
          break;
        default:
          require_once('login/login.php');
          break;
      }
    }
    break;
  case 'service':
    require_once('account/account.php');
    break;
  case 'error':
    require_once('error.php');
    break;
  default:
    require_once("error-404.php");
    break;
}
