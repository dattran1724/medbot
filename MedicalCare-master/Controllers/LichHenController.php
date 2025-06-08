<?php
require_once("Models/lichhen.php");
class LichHenController
{
    var $lichhen_model;

    public function __construct()
    {
        $this->lichhen_model = new Lichhen();
    }

    public function list()
    {
        if (isset($_GET['KH'])) {
            $data = $this->lichhen_model->lenlich($_GET['KH']);
        }  
        require_once('Views/index.php');
    }
    public function xemlichhen()
    {
        if ($_SESSION['isLogin'] == true) {
            $user_id = (isset($_SESSION['user']['userID']) && $_SESSION['user']['userID'] != null) ? $_SESSION['user']['userID'] : null;
            $data = $this->lichhen_model->xemlich($user_id);
        }
        // $data = $this->lichhen_model->xemlich(1);
        require_once('Views/index.php');
    }
    public function datlich()
    {
        $ngay = date_create($_POST['ngaysinh']);
        $name = $_POST['name'];
        $gender = $_POST['gender'];
        $phone2 = $_POST['phone2'];
        $ngaysinh = date_format($ngay,"d/m/Y");
        $tinh = $_POST['tinh'];
        $huyen = $_POST['huyen'];
        $xa = $_POST['xa'];
        $diachi = $_POST['diachi'];
        $lydo = $_POST['lydo'];

        if ($_SESSION['isLogin'] == true) {
            $check = $this->lichhen_model->check_benhnhan($_SESSION['user']['userID']) ;
            if (empty($check)) {
                $this->lichhen_model->add_benhnhan($_SESSION['user']['userID'],$name,$phone2,$gender, $ngaysinh, $tinh, $huyen, $xa, $diachi);
            }
            $bn = $this->lichhen_model->get_benhnhan($_SESSION['user']['userID']);

            $id_bs = $this->lichhen_model->get_id_bs($_GET['KH']);
            
            $this->lichhen_model->add_lichhen($bn[0]['id_bn'],$id_bs[0]['id_bs'],$_GET['KH'],$lydo) ;

        }
    }
}

?>