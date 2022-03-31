from controller import Robot

def bekle_zaman(sec):
    start_time = int(robot.getTime())
    while(start_time +sec > int(robot.getTime)):
        robot.step()

def duvarTakip(robot):
    timestep = 64

    l_motor = robot.getDevice("left wheel motor")
    r_motor = robot.getDevice("right wheel motor")
    l_motor.setPosition(float("inf"))
    r_motor.setPosition(float("inf"))
    l_motor.setVelocity(0.0)
    r_motor.setVelocity(0.0)
    max_uzaklık_kısıcı = 0.30
    max_uzaklık = 90
    # sensör tanılama
    yakinlikSensoru1 = robot.getDevice("ps1")
    yakinlikSensoru2 = robot.getDevice("ps2")
    yakinlikSensoru3 = robot.getDevice("ps3")
    yakinlikSensoru4 = robot.getDevice("ps4")
    yakinlikSensoru5 = robot.getDevice("ps5")
    yakinlikSensoru6 = robot.getDevice("ps6")
    yakinlikSensoru7 = robot.getDevice("ps7")
    yakinlikSensoru0 = robot.getDevice("ps0")

    # aktif edilmesi
    yakinlikSensoru0.enable(timestep)
    yakinlikSensoru1.enable(timestep)
    yakinlikSensoru2.enable(timestep)
    yakinlikSensoru3.enable(timestep)
    yakinlikSensoru4.enable(timestep)
    yakinlikSensoru5.enable(timestep)
    yakinlikSensoru6.enable(timestep)
    yakinlikSensoru7.enable(timestep)


    # Engel 8 türlü olabilir:
    # Tam önden --> sağa veya sola dön //
    # sağ üstten--> sol üste dön //
    # sol üstten --> sağ üste dön //
    # sağ tam yandan-->düz devam et //
    # sol tam yandan --> düz devam et //
    # eğer hem sağ üstten hemde sol üstten engel olursa--> geri vites
    #
    while robot.step(timestep) != -1:
        # önde engel olmadığı zaman için geçerli
        if (yakinlikSensoru7.getValue() < max_uzaklık and yakinlikSensoru0.getValue() < max_uzaklık):
            l_motor.setVelocity(5.0)
            r_motor.setVelocity(5.0)
            print("Önde Engel yok devam ediyorum.")
        # engel var ise
        else:
            # sağdan engel
            if (yakinlikSensoru5.getValue() > max_uzaklık):
                l_motor.setVelocity(5.0)
                r_motor.setVelocity(0.0)
                print("Tam sağda engel var, düz devam ediyorum.")
            # soldan engel
            if (yakinlikSensoru2.getValue() > max_uzaklık):
                l_motor.setVelocity(0.0)
                r_motor.setVelocity(5.0)
                print("Tam solda engel var. düz devam ediyorum.")

        # eğer engel sol üstten ise:
        if (yakinlikSensoru6.getValue() > max_uzaklık):
            l_motor.setVelocity(5.0)
            r_motor.setVelocity(0.0)
            print("Sol üstte engel var,devam ediyorum.")

        # Sol taraf tamamen aktif olursa:
        if (yakinlikSensoru7.getValue() > max_uzaklık and yakinlikSensoru6.getValue() > max_uzaklık and yakinlikSensoru5.getValue() > max_uzaklık and yakinlikSensoru4.getValue() > max_uzaklık):
            l_motor.setVelocity(5.0)
            r_motor.setVelocity(0.0)
            print("Sol taraf tamamen aktif ,sağdan devam ediyorum.")

            # eğer engel sağ üstten ise:
        if (yakinlikSensoru1.getValue() > max_uzaklık):
            l_motor.setVelocity(0.0)
            r_motor.setVelocity(5.0)
            print("sağ üstte engel var, devam ediyorum.")
        # eğer hem sağ üstten hemde sol üstten engel olursa
      #  if (yakinlikSensoru7.getValue() > max_uzaklık and yakinlikSensoru0.getValue() > max_uzaklık):
       #     l_motor.setVelocity(-5.0 * 0.2)
        #    r_motor.setVelocity(-5.0)

        # eğer hem sağ üstten hemde sol üstten engel olursa
        if (yakinlikSensoru7.getValue() > max_uzaklık and yakinlikSensoru0.getValue() > max_uzaklık):
            l_motor.setVelocity(-5.0 * 0.2)
            r_motor.setVelocity(-5.0)

        # eğer sağ ve üst tarafta engel olursa
        if (yakinlikSensoru2.getValue() > max_uzaklık and yakinlikSensoru0.getValue() > max_uzaklık and yakinlikSensoru1.getValue() > max_uzaklık ):
                l_motor.setVelocity(5.0*max_uzaklık_kısıcı)
                r_motor.setVelocity(0.0)

         # eğer sol ve üst tarafta engel olursa
        if (yakinlikSensoru7.getValue() > max_uzaklık and yakinlikSensoru5.getValue() > max_uzaklık and yakinlikSensoru6.getValue() > max_uzaklık ):
                 l_motor.setVelocity(0.0)
                 r_motor.setVelocity(5.0 * max_uzaklık_kısıcı)
        # Enter here exit cleanup code.


if __name__ == "__main__":
    robot = Robot()
    duvarTakip(robot)