<?php
require_once("Models/chuyenkhoa.php");
require_once("Models/chuyengia.php");
class HomeController
{
    var $home_model;
    var $chuyengia_model;
    var $chuyenkhoa_model;
    public function __construct()
    {
        $this->chuyenkhoa_model = new Chuyenkhoa();
        $this->chuyengia_model = new Chuyengia();
    }
    public function view(){
        $data_ck = $this->chuyenkhoa_model->limit(0,8);
        $data_bs = $this->chuyengia_model->limit(0, 6);
        require_once('Views/index.php');

    }

}

?>