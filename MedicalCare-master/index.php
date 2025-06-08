<?php
session_start();
$mod = isset($_GET['act']) ? $_GET['act'] : "home";
switch ($mod) {
  case 'home':
    require_once('Controllers/HomeController.php');
    $controller_obj = new HomeController();
    $controller_obj->view();
    break;
  case 'tintuc':
    require_once('Controllers/HomeController.php');
    $controller_obj = new HomeController();
    $controller_obj->view();
    break;
  case 'chuyengia':
    require_once('Controllers/ChuyenGiaController.php');
    $controller_obj = new ChuyenGiaController();
    $controller_obj->list();
    break;
  case 'chuyenkhoa':
    require_once('Controllers/ChuyenKhoaController.php');
    $controller_obj = new ChuyenKhoaController();
    $controller_obj->list();
    break;
  case 'chitietchuyenkhoa':
    require_once('Controllers/ChuyenKhoaController.php');
    $controller_obj = new ChuyenKhoaController();
    $controller_obj->chitietchuyenkhoa();
    break;
  case 'detail':
    require_once('Controllers/DetailController.php');
    $controller_obj = new DetailController();
    $controller_obj->list();
    break;
  case 'xemlichhen':
    require_once('Controllers/LichhenController.php');
    $controller_obj = new LichHenController();
    $controller_obj->xemlichhen();
    break;
  case 'lichhen':
    require_once('Controllers/LichHenController.php');
    $controller_obj = new LichHenController();

    $act = isset($_GET['xuli']) ? $_GET['xuli'] : "list";
    switch ($act) {
      case 'list':
        $controller_obj->list();
        break;
      case 'datlich':
        $controller_obj->datlich();
        break;
      default:
        $controller_obj->list();
        break;
    }
    break;
  case 'pharmacy':
    require_once('./Controllers/pharmacyController.php');
    $controller_obj = new PharmacyController();
    $controller_obj->list();
    break;
  case 'news':
    require_once('./Controllers/newsController.php');
    $controller_obj = new NewsController();
    $controller_obj->list();
    break;
  case 'news_detail':
    require_once('./Controllers/newsController.php');
    $controller_obj = new NewsController();
    $controller_obj->detail();
    break;
  case 'account':
    $act2 = isset($_GET['process']) ? $_GET['process'] : 'account';
    require_once('./Controllers/accountController.php');
    $controller_obj = new accountController();
    if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) {
      switch ($act2) {
        case 'account':
          $controller_obj->account();
          break;
        case 'logout':
          $controller_obj->logout();
          break;
        case 'update':
          $controller_obj->updateUser();
          break;
        case 'admin':
          $controller_obj->admin();
        default:
          header("Location: ?act=error");
          break;
      }
    } else {
      require_once('./Controllers/loginController.php');
      $controller_obj = new LoginController();
      switch ($act2) {
        case 'login':
          $controller_obj->login();
          break;
        case 'login_act':
          $controller_obj->login_action();
          break;
        case 'signup':
          $controller_obj->signup();
          break;
        case 'forgot':
          $controller_obj->forgot();
          break;
          case 'send':
            $controller_obj->sendEmail();
            break;
        case 'signup_act':
          $controller_obj->signup_action();
          break;
      }
    }
    break;
  case 'service':
    require_once('./Controllers/loginController.php');
    break;
  case 'admin':
    require_once('./Controllers/accountController.php');
    $controller_obj = new AccountController();
    $controller_obj->admin();
    break;
  case 'product-list':
    require_once('./Controllers/productListController.php');
    $controller_obj = new ListController();
    $controller_obj->list();
    break;
  case 'product-detail':
    require_once('./Controllers/productDetailController.php');
    $controller_obj = new DetailController();
    $act2 = isset($_GET['process']) ? $_GET['process'] : 'account';
    switch ($act2) {
      case 'add':
        require_once('./Controllers/cartController.php');
        $controller_obj = new CartController();
        $controller_obj->add();
        break;
      case 'delete':
        require_once('./Controllers/cartController.php');
        $controller_obj = new CartController();
        $controller_obj->delete();
        break;
      case 'update':
        require_once('./Controllers/cartController.php');
        $controller_obj = new CartController();
        $controller_obj->update();
        break;
      case 'list_order':
        require_once('./Controllers/orderController.php');
        $controller_obj = new OrderController();
        $controller_obj->listOrder();
        break;
      case 'make_order':
        require_once('./Controllers/orderController.php');
        $controller_obj = new OrderController();
        $controller_obj->makeOrder();
        break;
      case 'view_order':
        require_once('./Controllers/orderController.php');
        $controller_obj = new OrderController();
        $controller_obj->viewOrder();
        break;
      case 'order_detail':
        require_once('./Controllers/orderController.php');
        $controller_obj = new OrderController();
        $controller_obj->viewOrderDetail();
        break;
      case 'order_delete':
        require_once('./Controllers/orderController.php');
        $controller_obj = new OrderController();
        $controller_obj-> deleteOrder();
        break;
      default:
        $controller_obj->list();
        break;
    }
    break;
  case 'product-rating':
    require_once('./Controllers/productRatingController.php');
    $controller_obj = new RatingController();
    $controller_obj->submit();
    break;
  default:
    require_once('Controllers/HomeController.php');

    break;
}
