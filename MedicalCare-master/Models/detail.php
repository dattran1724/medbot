<?php 
    require_once("connection.php");
    // require_once(".php");
    class Detail {
        var $conn;
        function __construct()
        {
            $conn_obj = new connection();
            $this->conn = $conn_obj->conn;
        }
        function chitietbacsi($id){
            $query = "SELECT b.* FROM user as u, bacsi as b WHERE u.userID = b.userID and b.id_bs = $id" ;

            require("result.php");
            
            return $data ; 
        }
        function kehoach($id_bs){
            $query = "SELECT k.* ,s.ngay,s.gio FROM kehoach as k,status as s WHERE k.id_bs = $id_bs and k.id_status = s.id_status  ;" ;
            require("result.php");
            return $data ;
        }
        function ngay($id_bs){
            $query = "SELECT DISTINCT s.ngay FROM kehoach as k,status as s WHERE k.id_bs = $id_bs and k.id_status = s.id_status ;" ;
            require("result.php");
            return $data ;
        }
        function giotheongay($id_bs,$ngay){
            $query = "SELECT DISTINCT s.gio ,k.*  FROM kehoach as k,status as s WHERE k.id_bs = $id_bs and k.id_status = s.id_status and s.ngay = '$ngay'and k.trangthai=0 ;" ;
            require("result.php");
            return $data ;
        }
    }
?>