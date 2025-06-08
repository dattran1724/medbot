<?php
require_once("Models/chuyenkhoa.php");
require_once("Models/detail.php");
class ChuyenKhoaController
{
    var $chuyenkhoa_model;
    var $detail_model;
    public function __construct()
    {
       $this->chuyenkhoa_model = new Chuyenkhoa();
       $this->detail_model = new Detail();
    }
    
    public function list ()
    {   
        $id = isset($_GET['trang']) ? $_GET['trang'] : 1;
        $limit = 12;
        $start = ($id - 1) * $limit;
        $data = $this->chuyenkhoa_model->limit($start, $limit);
    
        $data_count = $this->chuyenkhoa_model->list();
        $data_tong  = count($data_count);

        require_once('Views/index.php');
    }
    public function chitietchuyenkhoa()
    {   
        if(isset($_GET['id'])){
            $data = $this->chuyenkhoa_model->chitietchuyenkhoa($_GET['id']);
            $data_bs = $this->chuyenkhoa_model->danhsachbacsi($_GET['id']);
            // $data_lich = $this->detail_model->kehoach($bs['id_bs']);
            // $ngay = $this->detail_model->ngay($bs['id_bs']);  

        }

        require_once('Views/index.php');
    }



}

?>