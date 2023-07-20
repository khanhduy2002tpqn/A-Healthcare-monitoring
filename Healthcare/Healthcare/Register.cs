using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MySql.Data.MySqlClient;


namespace WindowsFormsApp2
{
    public partial class Register : Form
    {
        public Register()
        {
            InitializeComponent();
        }
        MySqlConnectionStringBuilder conn_string = new MySqlConnectionStringBuilder();
        MySqlConnection MyCon;
        MySqlCommand cmd;
        private void Form2_Load(object sender, EventArgs e)
        {
            conn_string.Server = "172.31.9.10";
            conn_string.Port = 3306;
            conn_string.UserID = "root";
            conn_string.Password = "rasp1234";
            conn_string.Database = "diabetes";
            MyCon = new MySqlConnection(conn_string.ToString());
            MyCon.Open(); 
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton2.Checked)
            {
                textPre.Enabled = true;
                label5.Enabled = true;
            }
        }

        private void textBox6_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (textBox1.Text != string.Empty && Pass.Text != string.Empty && confPass.Text != string.Empty && textName.Text != string.Empty && textAge.Text != string.Empty && textPre.Text != string.Empty && radioButton1.Checked == true || radioButton2.Checked == true)
            { 
                var gender = "";
                if (radioButton1.Checked == true)
                {
                    gender = "0";
                }
                else
                {
                    gender = "1";
                }
                //generate random number for sensor id
                Random rnd = new Random();
                int sensor_id = rnd.Next(100000, 999999);
                //check if sensor id already exist
                cmd = new MySqlCommand("select * from user_client where Sensor_id ='" + sensor_id + "'", MyCon);
                MySqlDataReader dr = cmd.ExecuteReader();
                while (dr.Read())
                {
                    sensor_id = rnd.Next(100000, 999999);
                    cmd = new MySqlCommand("select * from user_client where Sensor_id ='" + sensor_id + "'", MyCon);
                    dr = cmd.ExecuteReader();
                }
                dr.Close();
                cmd = new MySqlCommand("insert into Sensor values('" + sensor_id + "','0','0','0','0','0','0','0')", MyCon);
                cmd.ExecuteNonQuery();
                if (Pass.Text == confPass.Text)
                {
                    
                    //insert into database
                    cmd = new MySqlCommand("insert into user_client values('" + textBox1.Text + "','" + textName.Text + "','" + Pass.Text + "','"+ sensor_id + "','"+ gender + "','" + textPre.Text + "','" + textAge.Text + "')", MyCon);
                    cmd.ExecuteNonQuery();
                    MessageBox.Show("Registration Successful" + "\n" + "Your Sensor ID is: " + sensor_id + "\n" + "Please remember your Sensor ID");                    
                    this.Hide();
                    Login login = new Login();
                    login.ShowDialog();

                }  
                
                else
                {
                    MessageBox.Show("Password and Confirm Password must be the same");
                }
            }
            else
            {
                MessageBox.Show("Please fill all the fields");
            }
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if(radioButton1.Checked == true)
            {
                textPre.Enabled = false;
                label5.Enabled = false;
                textPre.Text = "0";
            }
        }
    }
}
